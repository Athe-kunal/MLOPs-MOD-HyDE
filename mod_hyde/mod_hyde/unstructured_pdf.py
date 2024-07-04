from unstructured.partition.pdf import partition_pdf
from fire import Fire
import os
import json
os.makedirs("books_hi_res",exist_ok=True)

def pdf_text_extraction(filename:str,start_page:int,save_name:str):
    elements = partition_pdf(filename,strategy="hi_res")

    books_elem = []
    for _,elem in enumerate(elements):
        elem_dict = elem.to_dict()
        if elem_dict['type'] == 'PageBreak':continue
        page_num = elem_dict['metadata']['page_number']
        if page_num >= start_page:
            if elem_dict['type'] == 'NarrativeText':
                text = elem_dict['text']
                # num_words = len(text.split(" "))
                # if len(text)>=50 and num_words>=10:
                books_elem.append({
                    'text':text,
                    'page_num': page_num,
                    'coordinates': elem_dict['metadata']['coordinates']['points']
                })
    
    with open(f'books_hi_res/{save_name}.txt', 'w') as f:
        f.write(json.dumps(books_elem))
