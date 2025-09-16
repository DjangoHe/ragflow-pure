#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import copy
import re
import json

from api.db import LLMType
from api.db.services.llm_service import LLMBundle
from api.db import ParserType
from rag.nlp import rag_tokenizer, tokenize, tokenize_table, add_positions, bullets_category, title_frequency, tokenize_chunks
from deepdoc.parser import PdfParser, PlainParser
from deepdoc.parser.figure_parser import VisionFigureParser
from deepdoc.parser.flowchart_parser import FlowchartVisionParser
import numpy as np


class Pdf(PdfParser):
    def __init__(self):
        self.model_speciess = ParserType.PAPER.value
        super().__init__()

    def __call__(self, filename, binary=None, from_page=0,
                 to_page=100000, zoomin=3, callback=None, separate_tables_figures=False):

        from timeit import default_timer as timer
        start = timer()
        callback(msg="OCR started")
        self.__images__(
            filename if not binary else binary,
            zoomin,
            from_page,
            to_page,
            callback
        )
        callback(msg="OCR finished ({:.2f}s)".format(timer() - start))

        start = timer()
        self._layouts_rec(zoomin)
        callback(0.63, "Layout analysis ({:.2f}s)".format(timer() - start))
        logging.debug(f"layouts cost: {timer() - start}s")

        start = timer()
        self._table_transformer_job(zoomin)
        callback(0.68, "Table analysis ({:.2f}s)".format(timer() - start))

        start = timer()
        self._text_merge()
        figures = []
        if separate_tables_figures:
            tbls, figures = self._extract_table_figure(True, zoomin, True, True, True)
        else:
            tbls = self._extract_table_figure(True, zoomin, True, True)
        column_width = np.median([b["x1"] - b["x0"] for b in self.boxes])
        self._concat_downward()
        self._filter_forpages()
        callback(0.75, "Text merged ({:.2f}s)".format(timer() - start))

        # clean mess
        if column_width < self.page_images[0].size[0] / zoomin / 2:
            logging.debug("two_column................... {} {}".format(column_width,
                  self.page_images[0].size[0] / zoomin / 2))
            self.boxes = self.sort_X_by_page(self.boxes, column_width / 2)
        for b in self.boxes:
            b["text"] = re.sub(r"([\t 　]|\u3000){2,}", " ", b["text"].strip())

        def _begin(txt):
            return re.match(
                "[0-9. 一、i]*(introduction|abstract|摘要|引言|keywords|key words|关键词|background|背景|目录|前言|contents)",
                txt.lower().strip())

        if from_page > 0:
            return {
                "title": "",
                "authors": "",
                "abstract": "",
                "sections": [(b["text"] + self._line_tag(b, zoomin), b.get("layoutno", "")) for b in self.boxes if
                             re.match(r"(text|title)", b.get("layoutno", "text"))],
                "tables": tbls,
                "figures": figures
            }
        # get title and authors
        title = ""
        authors = []
        i = 0
        while i < min(32, len(self.boxes)-1):
            b = self.boxes[i]
            i += 1
            if b.get("layoutno", "").find("title") >= 0:
                title = b["text"]
                if _begin(title):
                    title = ""
                    break
                for j in range(3):
                    if _begin(self.boxes[i + j]["text"]):
                        break
                    authors.append(self.boxes[i + j]["text"])
                    break
                break
        # get abstract
        abstr = ""
        i = 0
        while i + 1 < min(32, len(self.boxes)):
            b = self.boxes[i]
            i += 1
            txt = b["text"].lower().strip()
            if re.match("(abstract|摘要)", txt):
                if len(txt.split()) > 32 or len(txt) > 64:
                    abstr = txt + self._line_tag(b, zoomin)
                    break
                txt = self.boxes[i]["text"].lower().strip()
                if len(txt.split()) > 32 or len(txt) > 64:
                    abstr = txt + self._line_tag(self.boxes[i], zoomin)
                i += 1
                break
        if not abstr:
            i = 0

        callback(
            0.8, "Page {}~{}: Text merging finished".format(
                from_page, min(
                    to_page, self.total_page)))
        for b in self.boxes:
            logging.debug("{} {}".format(b["text"], b.get("layoutno")))
        logging.debug("{}".format(tbls))

        return {
            "title": title,
            "authors": " ".join(authors),
            "abstract": abstr,
            "sections": [(b["text"] + self._line_tag(b, zoomin), b.get("layoutno", "")) for b in self.boxes[i:] if
                         re.match(r"(text|title)", b.get("layoutno", "text"))],
            "tables": tbls,
            "figures": figures
        }

def extract_is_flowchart(text):
    # 直接匹配 is_flowchart 的值
    match = re.search(r'is_flowchart\s*:\s*(true|false)', text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == 'true'
    return False

def merge_figures_and_boosted_figures(figures, boosted_figures):
    # 为每个boosted_figure提取is_flowchart
    for i, (image_data, positions) in enumerate(boosted_figures):
        text_description = image_data[1]  # 获取文本描述
        is_flowchart = extract_is_flowchart(text_description)
        boosted_figures[i] = (image_data[0], positions, is_flowchart)  # 添加is_flowchart字段

    # 合并到原始figures中
    merged_figures = []
    for i, (image_data, positions) in enumerate(figures):
        # 这里假设通过位置信息匹配，可以根据实际情况调整匹配逻辑
        is_flowchart = None
        for boosted_image_data, boosted_positions, boosted_is_flowchart in boosted_figures:
            if positions == boosted_positions:  # 匹配条件可以根据需要修改
                is_flowchart = boosted_is_flowchart
                break

        merged_figures.append((image_data, positions, is_flowchart))

    return merged_figures

def chunk(filename, binary=None, from_page=0, to_page=100000,
          lang="Chinese", callback=None, **kwargs):
    """
        Only pdf is supported.
        The abstract of the paper will be sliced as an entire chunk, and will not be sliced partly.
    """
    parser_config = kwargs.get(
        "parser_config", {
            "chunk_token_num": 512, "delimiter": "\n!?。；！？", "layout_recognize": "DeepDOC"})
    if re.search(r"\.pdf$", filename, re.IGNORECASE):
        if parser_config.get("layout_recognize", "DeepDOC") == "Plain Text":
            pdf_parser = PlainParser()
            paper = {
                "title": filename,
                "authors": " ",
                "abstract": "",
                "sections": pdf_parser(filename if not binary else binary, from_page=from_page, to_page=to_page)[0],
                "tables": []
            }
        else:
            pdf_parser = Pdf()

            try:
                vision_model = LLMBundle(kwargs["tenant_id"], LLMType.IMAGE2TEXT)
                callback(0.15, "Visual model detected. Attempting to enhance figure extraction...")
            except Exception:
                vision_model = None

            if vision_model:
                paper = pdf_parser(filename if not binary else binary,
                                   from_page=from_page, to_page=to_page, callback=callback,
                                   separate_tables_figures=True)
                figures = paper["figures"]
                callback(0.5, "Basic parsing complete. Proceeding with figure enhancement...")
                try:
                    pdf_vision_parser = VisionFigureParser(vision_model=vision_model, figures_data=figures, **kwargs)
                    boosted_figures = pdf_vision_parser(callback=callback)

                    # 提取is_flowchart，做意图识别
                    merged_figures = merge_figures_and_boosted_figures(figures, boosted_figures)

                    # 针对流程图的特殊处理
                    flowchart_parser = FlowchartVisionParser(
                        figures_data=merged_figures,
                        sam_model="vit_h",
                        sam_checkpoint_path="/root/model-weights/segment-anything/weights/sam_vit_h_4b8939.pth",
                        easyocr_model_path="/root/model-weights/easyocr"
                    )
                    flowchart_figures = flowchart_parser(callback=callback)
                    paper["tables"].extend(flowchart_figures)
                except Exception as e:
                    callback(0.6, f"Visual model error: {e}. Skipping figure parsing enhancement.")
                    paper["tables"].extend(figures)
            else:
                paper = pdf_parser(filename if not binary else binary,
                               from_page=from_page, to_page=to_page, callback=callback)
    else:
        raise NotImplementedError("file type not supported yet(pdf supported)")

    doc = {"docnm_kwd": filename, "authors_tks": rag_tokenizer.tokenize(paper["authors"]),
           "title_tks": rag_tokenizer.tokenize(paper["title"] if paper["title"] else filename)}
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
    doc["authors_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["authors_tks"])
    # is it English
    eng = lang.lower() == "english"  # pdf_parser.is_english
    logging.debug("It's English.....{}".format(eng))

    res = tokenize_table(paper["tables"], doc, eng)

    if paper["abstract"]:
        d = copy.deepcopy(doc)
        txt = pdf_parser.remove_tag(paper["abstract"])
        d["important_kwd"] = ["abstract", "总结", "概括", "summary", "summarize"]
        d["important_tks"] = " ".join(d["important_kwd"])
        d["image"], poss = pdf_parser.crop(
            paper["abstract"], need_position=True)
        add_positions(d, poss)
        tokenize(d, txt, eng)
        res.append(d)

    sorted_sections = paper["sections"]
    # set pivot using the most frequent type of title,
    # then merge between 2 pivot
    bull = bullets_category([txt for txt, _ in sorted_sections])
    most_level, levels = title_frequency(bull, sorted_sections)
    assert len(sorted_sections) == len(levels)
    sec_ids = []
    sid = 0
    for i, lvl in enumerate(levels):
        if lvl <= most_level and i > 0 and lvl != levels[i - 1]:
            sid += 1
        sec_ids.append(sid)
        logging.debug("{} {} {} {}".format(lvl, sorted_sections[i][0], most_level, sid))

    chunks = []
    last_sid = -2
    for (txt, _), sec_id in zip(sorted_sections, sec_ids):
        if sec_id == last_sid:
            if chunks:
                chunks[-1] += "\n" + txt
                continue
        chunks.append(txt)
        last_sid = sec_id
    res.extend(tokenize_chunks(chunks, doc, eng, pdf_parser))
    return res


"""
    readed = [0] * len(paper["lines"])
    # find colon firstly
    i = 0
    while i + 1 < len(paper["lines"]):
        txt = pdf_parser.remove_tag(paper["lines"][i][0])
        j = i
        if txt.strip("\n").strip()[-1] not in ":：":
            i += 1
            continue
        i += 1
        while i < len(paper["lines"]) and not paper["lines"][i][0]:
            i += 1
        if i >= len(paper["lines"]): break
        proj = [paper["lines"][i][0].strip()]
        i += 1
        while i < len(paper["lines"]) and paper["lines"][i][0].strip()[0] == proj[-1][0]:
            proj.append(paper["lines"][i])
            i += 1
        for k in range(j, i): readed[k] = True
        txt = txt[::-1]
        if eng:
            r = re.search(r"(.*?) ([\\.;?!]|$)", txt)
            txt = r.group(1)[::-1] if r else txt[::-1]
        else:
            r = re.search(r"(.*?) ([。？；！]|$)", txt)
            txt = r.group(1)[::-1] if r else txt[::-1]
        for p in proj:
            d = copy.deepcopy(doc)
            txt += "\n" + pdf_parser.remove_tag(p)
            d["image"], poss = pdf_parser.crop(p, need_position=True)
            add_positions(d, poss)
            tokenize(d, txt, eng)
            res.append(d)

    i = 0
    chunk = []
    tk_cnt = 0
    def add_chunk():
        nonlocal chunk, res, doc, pdf_parser, tk_cnt
        d = copy.deepcopy(doc)
        ck = "\n".join(chunk)
        tokenize(d, pdf_parser.remove_tag(ck), pdf_parser.is_english)
        d["image"], poss = pdf_parser.crop(ck, need_position=True)
        add_positions(d, poss)
        res.append(d)
        chunk = []
        tk_cnt = 0

    while i < len(paper["lines"]):
        if tk_cnt > 128:
            add_chunk()
        if readed[i]:
            i += 1
            continue
        readed[i] = True
        txt, layouts = paper["lines"][i]
        txt_ = pdf_parser.remove_tag(txt)
        i += 1
        cnt = num_tokens_from_string(txt_)
        if any([
            layouts.find("title") >= 0 and chunk,
            cnt + tk_cnt > 128 and tk_cnt > 32,
        ]):
            add_chunk()
            chunk = [txt]
            tk_cnt = cnt
        else:
            chunk.append(txt)
            tk_cnt += cnt

    if chunk: add_chunk()
    for i, d in enumerate(res):
        print(d)
        # d["image"].save(f"./logs/{i}.jpg")
    return res
"""

if __name__ == "__main__":
    import sys

    def dummy(prog=None, msg=""):
        pass
    chunk(sys.argv[1], callback=dummy)
