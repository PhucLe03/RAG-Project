# Parse code content to JSON
import json
import re
import pandas


change_json_pattern = r"\'[^\'\"]*?\'\: \'[^\'\"]*?\'"
change_code_json_pattern = r"\"code\"\: \'[^\'\"]*?\',"
href_pattern = r"href=\"[^\'\"]*?\""
code_sign = '"code": '

def json_reformat(code: str):
    code = code.replace('"Content-Type": "application/json"', "Content-Type: application/json")
    for match in re.finditer(change_json_pattern, code):
        found = match.group().replace("'", '"')
        code = code.replace(match.group(), found)
        code = code.replace("\\'next/link\\'", "next/link")
        code = code.replace("\\'next/server\\'", "next/server")
        code = code.replace("'code': ", '"code": ')
        code = code.replace('"code": \'', '"code": "')
        code = code.replace('"code":\'', '"code":"')
        code = code.replace('"Content-Type": "application/json"', "\'Content-Type\': \'application/json\'")
        code = code.replace("', \"language\":", "\", \"language\":")
        code = code.replace("'switcher': ", '"switcher": ')
    for match in re.finditer(href_pattern, code):
        found = match.group().replace("\"", "\'")
        code = code.replace(match.group(), found)
    for match in re.finditer(change_code_json_pattern, code):
        found = match.group().replace("\'", "\"")
        found = match.group().replace("'", "\"")
        found = match.group().replace('"code": \'', '"code": "')
        found = match.group().replace('"code":\'', '"code":"')
        found = match.group().replace("',", '",')
        code = code.replace(match.group(), found)
    return code

import ast

def parse_code_content(code_content: str):
    parsed_data = ast.literal_eval(code_content)
    return parsed_data
    # code_content = json_reformat(code_content)
    # # write to temp.json
    # code_content = code_content.replace("\n", "\\n")
    # # code_content = code_content.replace('\"', '\\"')
    # # code_content = code_content.replace("\'", "\\'")
    # code_content = code_content.replace("True", "true")
    # code_content = code_content.replace("False", "false")
    # with open("temp.json", "w") as f:
    #     f.write(code_content)
    # code_content_json = json.loads(code_content)
    # return code_content_json

def place_snippets_in_text(text_content: str, code_content_json: list) -> str:
    # return text_content + code_content_json
    code_template = "code_snippet_"
    text_content_add_snippets = text_content
    for i in range(len(code_content_json)):
        code_location = code_template + str(i+1)
        code_snippet = ""
        if code_content_json[i]['language']:
            code_snippet += code_content_json[i]['language']
        if code_content_json[i]['filename']:
            code_snippet += f" filename=\"{code_content_json[i]['filename']}\""
        if code_content_json[i]['switcher']:
            switcher = code_content_json[i]['switcher']
            if switcher:
                code_snippet += " switcher"
        code_snippet += "\n"
        code_snippet += code_content_json[i]['code']
        text_content_add_snippets = text_content_add_snippets.replace(code_location, code_snippet)
    return text_content_add_snippets

def get_retrieved_data(df: pandas.DataFrame, list_of_chunk_idx: list[int]) -> str:
    list_of_chunks = []
    for idx in list_of_chunk_idx:
        list_of_chunks.append(df.loc[idx])

    retrieved_data = ""

    for chunk in list_of_chunks:
        # print(chunk['code_content'])
        code_content_list = parse_code_content(chunk['code_content'])
        original_chunk = place_snippets_in_text(chunk['text_content'], code_content_list)
        retrieved_data += original_chunk
        retrieved_data += "\n\n"
    return retrieved_data

# print(retrieved_data)
import os

def save_to_file(filename, list_of_chunk_idx: list[int], context, question, answer) -> pandas.DataFrame:
    entry_ids = " ".join([str(chunk_id_str) for chunk_id_str in list_of_chunk_idx])
    # print(entry_ids)
    data_frame = pandas.DataFrame({"entry_ids": [entry_ids], 
                                   "context": [context], 
                                   "question": [question], 
                                   "answer": answer})
    if (os.path.exists(filename)):
        data_frame.to_csv(filename, index=False, mode='a', header=False)
    else:
        data_frame.to_csv(filename, index=False)
    return data_frame
