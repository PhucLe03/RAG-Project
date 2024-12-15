import os
import re
import csv
import frontmatter
from mistletoe.block_tokens import Document, Heading, CodeFence, List, Paragraph, ListItem
from mistletoe.span_tokens import RawText, InlineCode, Link
from milvus_model.hybrid import BGEM3EmbeddingFunction
import numpy as np

# Initialize the embedding function
embedding_function = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")

def extract_version_from_path(path):
    return os.path.basename(os.path.normpath(path))

def generate_entry_id(file_path):
    relative_path = os.path.relpath(file_path)
    entry_id = re.sub(r'[^a-zA-Z0-9]', '_', relative_path)
    return entry_id.lower()

def parse_markdown_sections(content):
    doc = Document.read(content)
    sections = []
    current_section = {"title": None, "content": "", "code_snippets": []}

    def extract_text_from_children(children):
        if not children:
            return ""
        text = ""
        for child in children:
            if isinstance(child, RawText):
                text += child.content
            elif isinstance(child, InlineCode):
                text += f"`{child.content}`"
            elif isinstance(child, Link):
                text += f"[{child.content or child.target}]({child.target})"
            elif hasattr(child, 'children') and child.children:
                text += extract_text_from_children(child.children)
        return text

    for token in doc.children:
        if isinstance(token, Heading):
            if current_section["title"] or current_section["content"]:
                sections.append(current_section)
            current_section = {
                "title": token.children[0].content if token.children else "",
                "content": "",
                "code_snippets": []
            }
        elif isinstance(token, CodeFence):
            filename = None
            switcher = False

            if token.arguments:
                match_filename = re.search(r'filename="([^"]+)"', token.arguments)
                if match_filename:
                    filename = match_filename.group(1)
                if 'switcher' in token.arguments:
                    switcher = True

            code_info = {
                "code": token.children[0].content if token.children else "",
                "language": token.language or "text",
                "filename": filename,
                "switcher": switcher
            }
            current_section["code_snippets"].append(code_info)
        elif isinstance(token, Paragraph):
            current_section["content"] += extract_text_from_children(token.children) + "\n"
        elif isinstance(token, List):
            for list_item in token.children:
                if isinstance(list_item, ListItem):
                    current_section["content"] += extract_text_from_children(list_item.children) + "\n"
        else:
            current_section["content"] += extract_text_from_children(token.children) + "\n"

    if current_section["title"] or current_section["content"]:
        sections.append(current_section)

    return sections

def process_sparse_vector(sparse_vector):
    return [(i, v) for i, v in zip(sparse_vector.indices, sparse_vector.data)]

def process_dense_vector(dense_vector):
    return dense_vector[0].tolist() if isinstance(dense_vector, list) and isinstance(dense_vector[0], np.ndarray) else dense_vector.tolist() if isinstance(dense_vector, np.ndarray) else dense_vector

def convert_to_serializable(data):
    if isinstance(data, np.ndarray):
        return process_dense_vector(data)
    elif hasattr(data, "indices") and hasattr(data, "data"):
        return process_sparse_vector(data)
    return data

def process_mdx_file(file_path, version):
    with open(file_path, 'r', encoding='utf-8') as f:
        post = frontmatter.load(f)

    if 'source' in post.metadata:
        return [], True

    entry_id = generate_entry_id(file_path)
    title = post.metadata.get('title', '')
    description = post.metadata.get('description', '')
    related_links = post.metadata.get('related', {}).get('links', [])
    nav_title = post.metadata.get('nav_title', '')

    sections = parse_markdown_sections(post.content)

    entries = []
    document_content = post.content
    for idx, section in enumerate(sections):
        if not section["title"]:
            document_content += section["content"]
            continue

        section_id = f"{entry_id}_section_{idx+1}"
        sparse_vector_input = f"{title} {section['title']}"
        section_entry = {
            "entry_id": section_id,
            "entry_type": "section",
            "parent_id": entry_id,
            "title": f"{title} - {section['title']}",
            "description": "",
            "metadata": {},
            "version": version,
            "text_content": section["content"],
            "code_content": None,
            "sparse_title_description": process_sparse_vector(
                embedding_function([sparse_vector_input])["sparse"] if sparse_vector_input else None
            ),
            "dense_text_content": process_dense_vector(
                embedding_function([section["content"]])["dense"] if section["content"] else None
            ),
            "dense_code_snippet": None,
            "tag": nav_title
        }
        entries.append(section_entry)

        for code_idx, code_snippet in enumerate(section["code_snippets"]):
            code_id = f"{section_id}_code_{code_idx+1}"
            sparse_vector_input = f"{code_snippet['language']} {code_snippet['filename']}"
            code_entry = {
                "entry_id": code_id,
                "entry_type": "code_snippet",
                "parent_id": section_id,
                "title": f"{section['title']} - code_{code_idx+1}",
                "description": "",
                "metadata": {
                    "programming_language": code_snippet["language"],
                    "filename": code_snippet["filename"],
                    "switcher": code_snippet["switcher"]
                },
                "version": version,
                "text_content": None,
                "code_content": code_snippet["code"],
                "sparse_title_description": process_sparse_vector(
                    embedding_function([sparse_vector_input])["sparse"] if sparse_vector_input else None
                ),
                "dense_text_content": None,
                "dense_code_snippet": process_dense_vector(
                    embedding_function([code_snippet["code"]])["dense"] if code_snippet["code"] else None
                ),
                "tag": nav_title
            }
            entries.append(code_entry)

    document_entry = {
        "entry_id": entry_id,
        "entry_type": "document",
        "parent_id": None,
        "title": title,
        "description": description,
        "metadata": {
            "related_links": related_links
        },
        "version": version,
        "text_content": document_content,
        "code_content": None,
        "sparse_title_description": process_sparse_vector(
            embedding_function([f"{title} {description}"])["sparse"] if title or description else None
        ),
        "dense_text_content": process_dense_vector(
            embedding_function([document_content])["dense"] if document_content else None
        ),
        "dense_code_snippet": None,
        "tag": nav_title
    }
    entries.insert(0, document_entry)

    return entries, False

def preprocess_directory(root_dir, version, batch_size=10):
    all_entries = []
    skipped_files = 0
    processed_files = 0
    batch_count = 0

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mdx'):
                file_path = os.path.join(subdir, file)
                print(f"Processing file: {file_path}")
                entries, skipped = process_mdx_file(file_path, version)
                if skipped:
                    skipped_files += 1
                    print(f"Skipped file: {file_path}")
                else:
                    processed_files += 1
                    all_entries.extend(entries)
                    print(f"Processed file: {file_path} ({len(entries)} entries)")

                if len(all_entries) >= batch_size:
                    batch_count += 1
                    batch_file = f"processed_entries_batch_{batch_count}.csv"
                    save_to_csv(all_entries, batch_file)
                    print(f"Saved batch {batch_count} to {batch_file}")
                    all_entries = []

    if all_entries:
        batch_count += 1
        batch_file = f"processed_entries_batch_{batch_count}.csv"
        save_to_csv(all_entries, batch_file)
        print(f"Saved batch {batch_count} to {batch_file}")

    print(f"Total files processed: {processed_files}")
    print(f"Total files skipped: {skipped_files}")
    print(f"Total batches saved: {batch_count}")

def save_to_csv(data, filename):
    keys = data[0].keys() if data else []
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for row in data:
            writer.writerow({k: convert_to_serializable(v) for k, v in row.items()})

# Example usage
root_directory = './15.1.0'
framework_version = '15.1.0'
preprocess_directory(root_directory, framework_version, batch_size=10)
