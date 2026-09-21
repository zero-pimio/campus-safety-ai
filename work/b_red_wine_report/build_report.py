from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_ROW_HEIGHT_RULE, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


PROJECT_ROOT = Path("/Users/wanwanzhu/yolo")
TEMPLATE = Path(
    "/Users/wanwanzhu/Library/Containers/com.tencent.xinWeChat/"
    "Data/Documents/xwechat_files/wxid_1q0qcco4cyfj22_d6f3/msg/file/"
    "2026-09/机器学习大作业模板.docx"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DOCX = OUTPUT_DIR / "基于机器学习的红酒质量预测研究报告.docx"
FIGURE_DIR = PROJECT_ROOT / "work" / "b_red_wine_figures"

CHINESE_FONT = "Heiti SC"
SERIF_FONT = "Times New Roman"
CODE_FONT = "Menlo"
BLACK = RGBColor(0, 0, 0)
DARK_BLUE = "1F4E78"
LIGHT_BLUE = "D9EAF7"
LIGHT_GRAY = "F2F2F2"
GRID_GRAY = "BFBFBF"


def set_run_font(run, font_name=CHINESE_FONT, size=10.5, bold=False, italic=False, color=BLACK):
    run.font.name = font_name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:ascii"), font_name)
    rfonts.set(qn("w:hAnsi"), font_name)
    rfonts.set(qn("w:eastAsia"), font_name)


def clear_paragraph(paragraph):
    for child in list(paragraph._p):
        if child.tag != qn("w:pPr"):
            paragraph._p.remove(child)


def write_paragraph(
    paragraph,
    text,
    *,
    size=10.5,
    bold=False,
    align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    font_name=CHINESE_FONT,
    first_indent=True,
    line_spacing=1.5,
    before=0,
    after=5,
):
    clear_paragraph(paragraph)
    paragraph.alignment = align
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line_spacing
    if first_indent:
        fmt.first_line_indent = Cm(0.74)
    else:
        fmt.first_line_indent = Cm(0)
    run = paragraph.add_run(text)
    set_run_font(run, font_name=font_name, size=size, bold=bold)
    return paragraph


def style_heading(paragraph, text, level=1, page_break_before=False):
    clear_paragraph(paragraph)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if level == 1 else WD_ALIGN_PARAGRAPH.LEFT
    fmt = paragraph.paragraph_format
    # LibreOffice treats a literal <w:pageBreakBefore w:val="0"/> from the
    # template as a break. Remove that element unless a break is actually wanted.
    fmt.page_break_before = True if page_break_before else None
    fmt.keep_with_next = True
    fmt.space_before = Pt(0 if level == 1 else 8)
    fmt.space_after = Pt(10 if level == 1 else 5)
    fmt.line_spacing = 1.25
    run = paragraph.add_run(text)
    if level == 1:
        set_run_font(run, size=16, bold=True)
    elif level == 2:
        set_run_font(run, size=14, bold=True)
    else:
        set_run_font(run, size=12, bold=True)
    return paragraph


def set_cell_text(cell, text, *, bold=False, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, shade=None):
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    paragraph = cell.paragraphs[0]
    write_paragraph(
        paragraph,
        text,
        size=size,
        bold=bold,
        align=align,
        first_indent=False,
        line_spacing=1.15,
        before=0,
        after=0,
    )
    for paragraph in cell.paragraphs[1:]:
        clear_paragraph(paragraph)
    if shade:
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = tc_pr.find(qn("w:shd"))
        if shd is None:
            shd = OxmlElement("w:shd")
            tc_pr.append(shd)
        shd.set(qn("w:fill"), shade)


def set_cell_margins(cell, top=100, start=120, bottom=100, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin_name, margin_value in (
        ("top", top),
        ("start", start),
        ("bottom", bottom),
        ("end", end),
    ):
        node = tc_mar.find(qn(f"w:{margin_name}"))
        if node is None:
            node = OxmlElement(f"w:{margin_name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(margin_value))
        node.set(qn("w:type"), "dxa")


def set_cell_border(cell, color=GRID_GRAY, size="6"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def format_table(table, header=True, font_size=9.2):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for r_i, row in enumerate(table.rows):
        row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
        for cell in row.cells:
            set_cell_margins(cell)
            set_cell_border(cell)
            shade = DARK_BLUE if header and r_i == 0 else (LIGHT_BLUE if r_i % 2 == 0 else "FFFFFF")
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.1
                for run in paragraph.runs:
                    set_run_font(
                        run,
                        size=font_size,
                        bold=(header and r_i == 0),
                        color=RGBColor(255, 255, 255) if header and r_i == 0 else BLACK,
                    )
            tc_pr = cell._tc.get_or_add_tcPr()
            shd = tc_pr.find(qn("w:shd"))
            if shd is None:
                shd = OxmlElement("w:shd")
                tc_pr.append(shd)
            shd.set(qn("w:fill"), shade)
    if header:
        set_repeat_table_header(table.rows[0])


def insert_before(anchor, text="", *, kind="body", page_break_before=False):
    paragraph = anchor.insert_paragraph_before()
    if kind == "h1":
        style_heading(paragraph, text, level=1, page_break_before=page_break_before)
    elif kind == "h2":
        style_heading(paragraph, text, level=2, page_break_before=page_break_before)
    elif kind == "h3":
        style_heading(paragraph, text, level=3, page_break_before=page_break_before)
    elif kind == "caption":
        write_paragraph(
            paragraph,
            text,
            size=9,
            align=WD_ALIGN_PARAGRAPH.CENTER,
            first_indent=False,
            line_spacing=1.1,
            before=2,
            after=7,
        )
        paragraph.paragraph_format.keep_with_next = False
    elif kind == "code":
        write_paragraph(
            paragraph,
            text,
            size=8.2,
            align=WD_ALIGN_PARAGRAPH.LEFT,
            font_name=CODE_FONT,
            first_indent=False,
            line_spacing=1.0,
            before=0,
            after=0,
        )
    else:
        write_paragraph(
            paragraph,
            text,
            size=10.5,
            align=WD_ALIGN_PARAGRAPH.JUSTIFY,
            first_indent=True,
            line_spacing=1.5,
            before=0,
            after=5,
        )
        paragraph.paragraph_format.page_break_before = page_break_before
    return paragraph


def add_native_equation(anchor, equation):
    paragraph = anchor.insert_paragraph_before()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(6)
    omath = OxmlElement("m:oMath")
    math_run = OxmlElement("m:r")
    math_props = OxmlElement("m:rPr")
    math_size = OxmlElement("m:sz")
    math_size.set(qn("m:val"), "22")
    math_props.append(math_size)
    math_run.append(math_props)
    math_text = OxmlElement("m:t")
    math_text.text = equation
    math_run.append(math_text)
    omath.append(math_run)
    paragraph._p.append(omath)
    return paragraph


def add_report_table(anchor, headers, rows, widths, font_size=9.2):
    table = anchor._parent.add_table(rows=1, cols=len(headers), width=Cm(sum(widths)))
    table_element = table._tbl
    anchor._p.addprevious(table_element)
    for i, text in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], str(text), bold=True, size=font_size)
        table.rows[0].cells[i].width = Cm(widths[i])
    for row_values in rows:
        cells = table.add_row().cells
        for i, text in enumerate(row_values):
            set_cell_text(cells[i], str(text), size=font_size)
            cells[i].width = Cm(widths[i])
    format_table(table, header=True, font_size=font_size)
    caption = insert_before(anchor, "", kind="caption")
    return table, caption


def add_figure(anchor, figure_name, caption, width_cm=12.5):
    image_path = FIGURE_DIR / figure_name
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    paragraph = anchor.insert_paragraph_before()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.keep_together = True
    paragraph.add_run().add_picture(str(image_path), width=Cm(width_cm))
    insert_before(anchor, caption, kind="caption")


def clear_footer(section, with_page_number=False):
    footer = section.footer
    for paragraph in footer.paragraphs:
        clear_paragraph(paragraph)
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    if with_page_number:
        run = paragraph.add_run("— ")
        set_run_font(run, font_name=SERIF_FONT, size=9)
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), "PAGE")
        paragraph._p.append(field)
        run = paragraph.add_run(" —")
        set_run_font(run, font_name=SERIF_FONT, size=9)


def clear_header(section):
    header = section.header
    for paragraph in header.paragraphs:
        clear_paragraph(paragraph)


def main():
    if not TEMPLATE.exists():
        raise FileNotFoundError(TEMPLATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    document = Document(TEMPLATE)
    source = list(document.paragraphs)
    if len(source) < 139 or len(document.tables) != 2:
        raise RuntimeError("报告模板结构与核验记录不一致")
    # Capture the template-owned feedback table before inserting report tables;
    # document.tables is live and its indexes change after insertion.
    feedback_table = document.tables[1]
    # Remove the template-only section break between contents and abstract. It
    # produces an otherwise empty rendered page; the abstract heading below has
    # an explicit page break and therefore supplies the intended page boundary.
    abstract_section_break = source[69]._p.pPr.sectPr
    if abstract_section_break is None:
        raise RuntimeError("模板目录后的分节符缺失")
    source[69]._p.pPr.remove(abstract_section_break)

    for section_index, section in enumerate(document.sections):
        clear_header(section)
        clear_footer(section, with_page_number=section_index in (1, 2, 3))

    # Cover page
    for index in range(0, 28):
        clear_paragraph(source[index])
    # The template has several tall blank cover paragraphs after the metadata
    # table. Compact only those blanks so the date remains on the cover page.
    for index in range(8, 14):
        fmt = source[index].paragraph_format
        fmt.space_before = Pt(0)
        fmt.space_after = Pt(0)
        fmt.line_spacing = 1.0
        fmt.first_line_indent = Cm(0)
    write_paragraph(
        source[4],
        "机器学习课程大作业报告",
        size=24,
        bold=True,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        first_indent=False,
        line_spacing=1.0,
        before=0,
        after=10,
    )
    cover_table = document.tables[0]
    cover_values = [
        ("题    目", "基于机器学习的红酒质量预测研究"),
        ("学    院", ""),
        ("专    业", ""),
        ("姓    名", ""),
        ("学    号", ""),
        ("任课教师", ""),
        ("成    绩", ""),
    ]
    for row, (label, value) in zip(cover_table.rows, cover_values):
        set_cell_text(row.cells[0], label, bold=True, size=11, shade=LIGHT_GRAY)
        set_cell_text(row.cells[1], value, size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
        for cell in row.cells:
            set_cell_margins(cell, top=150, start=140, bottom=150, end=140)
            set_cell_border(cell, color="808080", size="8")
    cover_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # The template places its date after multiple cover spacers; leaving it
    # empty avoids an isolated date line above the table of contents.
    clear_paragraph(source[14])
    source[14].paragraph_format.space_before = Pt(0)
    source[14].paragraph_format.space_after = Pt(0)

    # Table of contents
    for index in range(28, 71):
        clear_paragraph(source[index])
    write_paragraph(
        source[28],
        "目 录",
        size=16,
        bold=True,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        first_indent=False,
        line_spacing=1.0,
        before=0,
        after=18,
    )
    toc_entries = [
        "摘 要……………………………………………………………………………3",
        "ABSTRACT………………………………………………………………………4",
        "第一章 绪论……………………………………………………………………5",
        "第二章 数据集与方法…………………………………………………………6",
        "第三章 实验设计与模型优化…………………………………………………8",
        "第四章 实验结果与分析………………………………………………………10",
        "第五章 结论与展望……………………………………………………………12",
        "附录A 关键源代码……………………………………………………………13",
        "参考文献………………………………………………………………………14",
    ]
    for paragraph, entry in zip(source[31:40], toc_entries):
        write_paragraph(
            paragraph,
            entry,
            size=11,
            align=WD_ALIGN_PARAGRAPH.LEFT,
            first_indent=False,
            line_spacing=1.35,
            before=0,
            after=3,
        )

    # Chinese and English abstracts
    for index in range(70, 103):
        clear_paragraph(source[index])
    # The removed section break already leaves the abstract on the next page.
    style_heading(source[70], "摘  要", level=1)
    chinese_abstract = (
        "红酒质量评价通常依赖品酒师的感官评分，而理化指标能够在生产与质检环节较早获得。"
        "为探讨基于机器学习的自动预测方法，本文使用红酒质量数据集中的1599个样本，以11项理化指标为输入，"
        "以quality等级为输出，严格按题意将前1300条样本划分为训练集、后299条样本划分为测试集。"
        "实验在训练集内完成标准化、特征选择、主成分分析、多模型比较和5折分层交叉验证；比较逻辑回归、"
        "支持向量机、随机森林、极端随机树和直方图梯度提升等方法。针对类别不均衡问题，模型选择以macro-F1为主，"
        "并同时报告accuracy、balanced accuracy、weighted-F1和平均绝对误差。"
        "最终采用SelectKBest与ExtraTrees组成的流水线，在48组参数上进行网格搜索，最优交叉验证macro-F1为0.4529。"
        "最终模型在299个测试样本上的accuracy为0.5518、macro-F1为0.2776、MAE为0.5318。结果表明，"
        "模型对5级和6级这一主要样本区间具有较稳定的预测能力，但对数量极少的3级和8级样本识别仍受限。"
        "本文给出了从数据核验、特征工程到最终预测的可复现实验流程，并据此分析方法的适用边界。"
    )
    write_paragraph(source[73], chinese_abstract, size=10.5, line_spacing=1.5, after=8)
    write_paragraph(
        source[80],
        "关键词：红酒质量预测  机器学习  特征选择  极端随机树  交叉验证",
        size=10.5,
        bold=True,
        first_indent=False,
        line_spacing=1.5,
        after=0,
    )
    style_heading(source[86], "ABSTRACT", level=1)
    english_abstract = (
        "This study develops a reproducible machine-learning workflow for red wine quality prediction "
        "from physicochemical measurements. The dataset contains 1,599 samples and eleven input variables. "
        "Following the assignment specification, the first 1,300 samples are used for training and the remaining "
        "299 samples are held out for final testing. All preprocessing, feature selection, principal component "
        "analysis, model comparison, and hyperparameter search are conducted on the training set only. "
        "Logistic regression, support vector machine, random forest, ExtraTrees, and histogram gradient boosting "
        "are compared with five-fold stratified cross-validation. Because the quality classes are imbalanced, "
        "macro-F1 is used as the primary selection metric alongside accuracy, balanced accuracy, weighted-F1, and MAE. "
        "The selected SelectKBest plus ExtraTrees pipeline obtains a best cross-validation macro-F1 of 0.4529. "
        "On the held-out test set, it achieves an accuracy of 0.5518, a macro-F1 of 0.2776, and an MAE of 0.5318. "
        "The results show reasonable discrimination for the dominant quality levels while also revealing the difficulty "
        "of predicting very rare quality scores."
    )
    write_paragraph(
        source[89],
        english_abstract,
        size=10.5,
        font_name=SERIF_FONT,
        line_spacing=1.5,
        after=8,
    )
    write_paragraph(
        source[95],
        "Key words: red wine quality prediction; machine learning; feature selection; ExtraTrees; cross-validation",
        size=10.5,
        bold=True,
        font_name=SERIF_FONT,
        first_indent=False,
        line_spacing=1.5,
        after=0,
    )

    # Remove source body placeholders but keep its section-break paragraph as the insertion anchor.
    for paragraph in source[103:123]:
        parent = paragraph._p.getparent()
        if parent is not None:
            parent.remove(paragraph._p)
    body_anchor = source[123]

    # Main page 1: introduction
    insert_before(body_anchor, "第一章 绪论", kind="h1", page_break_before=True)
    insert_before(body_anchor, "1.1 研究背景与意义", kind="h2")
    insert_before(
        body_anchor,
        "红酒的风味与品质受原料、发酵、储存等多种因素影响。传统质量评价主要依赖感官评定，"
        "具有直观有效的优点，但也会受到评价成本、人员经验和主观差异的影响。酒液的酸度、挥发酸、"
        "硫化物、密度、pH、硫酸盐和酒精度等理化指标则易于在生产检测中获取。若能从这些客观指标中学习"
        "质量等级与特征之间的关系，就可以为筛查、过程监控和辅助品控提供定量参考。"
    )
    insert_before(
        body_anchor,
        "本题数据来自葡萄牙绿酒的红酒子集，quality由至少3位专家评分的中位数形成[1]。"
        "该任务既可以视作有序回归问题，也可以视作多分类问题。考虑到题目要求使用predict和argmax进行预测，"
        "本文将quality的离散等级3至8建模为六分类任务，并以类别概率最大的等级作为输出。"
    )
    insert_before(body_anchor, "1.2 问题描述与研究目标", kind="h2")
    insert_before(
        body_anchor,
        "研究目标是在不改变样本顺序的前提下，使用前1300条记录训练模型，并对后299条记录进行独立测试。"
        "实验首先核验数据的维度、字段类型和缺失情况；随后使用训练集完成模型选择和超参数优化；最后仅在测试集上"
        "给出一次性预测与评价。该设置避免测试集信息进入建模过程，使最终指标能够反映模型面对未参与训练样本时的表现。"
    )
    insert_before(body_anchor, "1.3 研究路线", kind="h2")
    insert_before(
        body_anchor,
        "本文遵循“数据核验—探索分析—特征处理—交叉验证—参数优化—测试评估”的路线。"
        "其中，所有会从数据学习参数的步骤均放入Pipeline或交叉验证折内执行，防止在训练集内部产生数据泄漏；"
        "同时保留多数类基线，以判断复杂模型是否真正优于简单的类别频率预测。"
    )

    # Main page 2: data
    insert_before(body_anchor, "第二章 数据集与方法", kind="h1", page_break_before=True)
    insert_before(body_anchor, "2.1 数据集构成与质量核验", kind="h2")
    insert_before(
        body_anchor,
        "数据文件采用分号分隔，共1599条样本和12列字段，其中11列为连续型理化指标，quality为0至10分制中的离散感官等级。"
        "本次数据实际出现的quality等级为3、4、5、6、7和8。读取后检验未发现缺失值；数据中有240条重复记录，"
        "但为严格维持题目规定的1599条样本与固定前后切分，实验未删除或合并重复记录。"
    )
    table, caption = add_report_table(
        body_anchor,
        ["项目", "实际结果", "处理方式"],
        [
            ["样本总数", "1599", "保留全部样本"],
            ["输入特征", "11项理化指标", "连续数值变量"],
            ["标签", "quality，取值3至8", "六分类建模"],
            ["缺失值", "0", "无需插补"],
            ["重复记录", "240", "保留以符合固定切分"],
            ["训练集/测试集", "1300 / 299", "按原始顺序切分"],
        ],
        [3.6, 4.4, 7.0],
    )
    caption.text = "表2-1 数据集核验与处理结果"
    write_paragraph(
        caption,
        "表2-1 数据集核验与处理结果",
        size=9,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        first_indent=False,
        line_spacing=1.1,
        before=2,
        after=7,
    )
    insert_before(body_anchor, "2.2 标签分布与类别不均衡", kind="h2")
    insert_before(
        body_anchor,
        "质量等级分布高度集中在5级和6级：两类合计1319条，占全部样本的82.49%。3级和8级分别只有10条和18条。"
        "因此，单独使用accuracy容易掩盖少数类别的误判。本文采用分层交叉验证保持每折的类别比例，并把macro-F1作为"
        "超参数选择主指标，使每一个质量等级的F1分数都具有相同权重。"
    )
    add_figure(body_anchor, "01_quality_distribution.png", "图2-1 红酒质量等级分布", width_cm=10.5)

    # Main page 3: methods and equations
    insert_before(body_anchor, "2.3 特征选择、特征提取与候选模型", kind="h2", page_break_before=True)
    insert_before(
        body_anchor,
        "特征选择使用ANOVA F检验，对每个理化指标与多分类标签之间的组间差异进行打分。"
        "F分数越高，说明该指标在不同质量等级之间的均值差异越明显。训练集上的结果显示，alcohol和volatile acidity"
        "最具区分度；残糖residual sugar的p值为0.0620，在当前训练集上未达到0.05显著性水平。最终网格搜索选择10个特征，"
        "仅排除residual sugar。"
    )
    selected_table, caption = add_report_table(
        body_anchor,
        ["类别", "特征"],
        [
            ["最终保留", "fixed acidity，volatile acidity，citric acid，chlorides"],
            ["最终保留", "free sulfur dioxide，total sulfur dioxide，density，pH"],
            ["最终保留", "sulphates，alcohol"],
            ["最终排除", "residual sugar"],
        ],
        [3.0, 12.0],
        font_size=8.7,
    )
    write_paragraph(
        caption,
        "表2-2 SelectKBest最终特征集合",
        size=9,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        first_indent=False,
        line_spacing=1.1,
        before=2,
        after=7,
    )
    insert_before(
        body_anchor,
        "作为特征提取对照，本文还对标准化后的特征进行主成分分析（PCA）。训练集上保留95%累计解释方差需要9个主成分。"
        "PCA+逻辑回归被纳入模型比较，但最终没有作为最优方案；原因是PCA压缩后的成分可解释性较弱，且其macro-F1低于树模型。"
    )
    insert_before(body_anchor, "2.4 评价指标与验证方式", kind="h2")
    insert_before(
        body_anchor,
        "设测试或验证样本总数为N，真实标签为yᵢ，预测标签为ŷᵢ。accuracy用于衡量整体正确率；"
        "macro-F1先分别计算每个类别的F1后再求平均，更适合观察不均衡类别的总体识别效果；MAE则衡量预测等级与真实等级之间的平均绝对偏差。"
    )
    add_native_equation(body_anchor, "Accuracy = (1 / N) Σ I(yᵢ = ŷᵢ)")
    add_native_equation(body_anchor, "F₁ = 2PR / (P + R)      MAE = (1 / N) Σ |yᵢ - ŷᵢ|")
    insert_before(
        body_anchor,
        "交叉验证采用5折StratifiedKFold，并固定random_state为42。每一折都重新拟合标准化、SelectKBest和PCA等变换，"
        "再在验证部分评分。这样可避免将整套训练数据的统计特征提前暴露给某个验证折。"
    )

    # Main page 4: model comparison
    insert_before(body_anchor, "第三章 实验设计与模型优化", kind="h1", page_break_before=True)
    insert_before(body_anchor, "3.1 实验环境与可复现设置", kind="h2")
    insert_before(
        body_anchor,
        "实验使用Python 3.11、pandas 3.0.5和scikit-learn 1.9.1完成。所有随机森林和ExtraTrees模型的random_state固定为42，"
        "并将n_jobs设为1，以避免并行环境差异影响实验记录。Notebook保留了数据读取、图表生成、模型训练、网格搜索和测试预测的原始输出。"
    )
    insert_before(body_anchor, "3.2 训练集内模型比较", kind="h2")
    insert_before(
        body_anchor,
        "候选模型包括多数类基线、逻辑回归、逻辑回归+SelectKBest、PCA+逻辑回归、RBF核SVC、随机森林、ExtraTrees和"
        "直方图梯度提升。下表列出训练集5折交叉验证均值。基线准确率为0.4200，说明仅预测多数类别会造成表面正确率；"
        "树模型在accuracy、weighted-F1和MAE方面具有优势，而直方图梯度提升在未调参模型中取得最高macro-F1。"
    )
    cv_table, caption = add_report_table(
        body_anchor,
        ["模型", "Accuracy", "Balanced Acc.", "Macro-F1", "Weighted-F1", "MAE"],
        [
            ["HistGradientBoosting", "0.6862", "0.3753", "0.3875", "0.6719", "0.3523"],
            ["ExtraTrees", "0.7092", "0.3754", "0.3839", "0.6921", "0.3254"],
            ["RandomForest", "0.7077", "0.3708", "0.3832", "0.6891", "0.3238"],
            ["SVC (RBF)", "0.5323", "0.3647", "0.3233", "0.5487", "0.5531"],
            ["LogisticRegression", "0.4277", "0.3897", "0.2827", "0.4696", "0.7685"],
            ["Majority baseline", "0.4200", "0.1667", "0.0986", "0.2485", "0.7446"],
        ],
        [4.1, 2.1, 2.7, 2.0, 2.3, 1.6],
        font_size=8.2,
    )
    write_paragraph(
        caption,
        "表3-1 候选模型的5折交叉验证结果",
        size=9,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        first_indent=False,
        line_spacing=1.1,
        before=2,
        after=5,
    )
    insert_before(
        body_anchor,
        "综合可解释性、特征选择能力和对不均衡指标的优化空间，后续选择ExtraTrees与SelectKBest的组合进行调参。"
        "这不是使用测试集做出的选择，而是基于训练集交叉验证的模型选择决策。"
    )

    # Main page 5: tuning
    insert_before(body_anchor, "3.3 网格搜索与超参数选择", kind="h2", page_break_before=True)
    insert_before(
        body_anchor,
        "调参流水线由SelectKBest和ExtraTreesClassifier组成。搜索范围包括保留特征数k、树数量、最大深度、叶节点最小样本数、"
        "最大特征比例和类别权重，共形成48组配置。GridSearchCV以macro-F1为refit指标，仍使用前述5折分层划分。"
    )
    param_table, caption = add_report_table(
        body_anchor,
        ["参数", "搜索候选", "最终选择"],
        [
            ["select__k", "8，10，11", "10"],
            ["n_estimators", "200", "200"],
            ["max_depth", "None，16", "16"],
            ["min_samples_leaf", "2，4", "4"],
            ["max_features", "sqrt，0.8", "0.8"],
            ["class_weight", "None，balanced", "balanced"],
        ],
        [4.3, 5.2, 4.7],
    )
    write_paragraph(
        caption,
        "表3-2 ExtraTrees网格搜索范围与最优参数",
        size=9,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        first_indent=False,
        line_spacing=1.1,
        before=2,
        after=7,
    )
    insert_before(
        body_anchor,
        "最优配置的5折macro-F1为0.4529，balanced accuracy为0.4874，accuracy为0.6600，weighted-F1为0.6634，"
        "交叉验证MAE为0.3985。与未调参ExtraTrees的macro-F1 0.3839相比，调参后的流水线提升0.0690，"
        "说明在类别权重、树深度和叶节点规模上进行约束有助于兼顾多数类与少数类。"
    )
    add_figure(body_anchor, "05_model_cv_comparison.png", "图3-1 候选模型在训练集上的交叉验证比较", width_cm=13.2)

    # Main page 6: feature analysis
    insert_before(body_anchor, "第四章 实验结果与分析", kind="h1", page_break_before=True)
    insert_before(body_anchor, "4.1 特征区分度与主成分分析", kind="h2")
    insert_before(
        body_anchor,
        "ANOVA F检验显示，alcohol的F分数为103.1595，明显高于其他变量；volatile acidity的F分数为49.1978，"
        "total sulfur dioxide为23.1780。该排序表明酒精度、挥发酸和总二氧化硫在训练样本不同质量等级之间具有较强区分度。"
        "但F检验只刻画单变量与标签的关系，最终模型仍需要通过树的分裂组合刻画多个指标之间的非线性关系。"
    )
    add_figure(body_anchor, "03_feature_selection_scores.png", "图4-1 训练集上各理化指标的ANOVA F分数", width_cm=11.8)
    insert_before(
        body_anchor,
        "标准化后的PCA结果表明，9个主成分可达到95%的累计解释方差。PCA有助于降低维度和缓解相关性，"
        "但在本实验中PCA+逻辑回归的5折macro-F1为0.2775，低于树模型；因此最终模型保留原始可解释特征。"
    )
    add_figure(body_anchor, "04_pca_explained_variance.png", "图4-2 训练集PCA累计解释方差", width_cm=10.8)

    # Main page 7: final test results
    insert_before(body_anchor, "4.2 最终测试集预测结果", kind="h2", page_break_before=True)
    insert_before(
        body_anchor,
        "确定最优参数后，使用完整的1300条训练数据重新拟合Pipeline，并对299条测试数据调用predict。"
        "同时调用predict_proba得到各等级概率，并以argmax选择概率最大的类别；两种方式得到的标签已在Notebook中通过断言核对一致。"
        "测试集在调参和模型选择过程中从未使用，因此以下结果是独立测试结果。"
    )
    metric_table, caption = add_report_table(
        body_anchor,
        ["指标", "测试集结果", "解释"],
        [
            ["Accuracy", "0.5518", "299条样本中整体正确率"],
            ["Balanced accuracy", "0.3006", "各类别召回率的平均值"],
            ["Macro-F1", "0.2776", "各类别F1的等权平均"],
            ["Weighted-F1", "0.5634", "按类别样本数加权的F1"],
            ["MAE", "0.5318", "预测等级与真实等级的平均绝对偏差"],
        ],
        [4.4, 3.2, 6.6],
    )
    write_paragraph(
        caption,
        "表4-1 最优模型在独立测试集上的评价指标",
        size=9,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        first_indent=False,
        line_spacing=1.1,
        before=2,
        after=7,
    )
    insert_before(
        body_anchor,
        "从逐类结果看，5级的F1为0.6768、6级的F1为0.5508，是模型主要有效的等级区间；7级样本的召回率为0.4286，"
        "但精确率为0.2308。3级和8级在测试集中分别只有4条和3条，最终未被正确识别。该现象与训练集中极端等级样本稀少相一致，"
        "也解释了accuracy高于macro-F1的原因。"
    )
    add_figure(body_anchor, "06_test_confusion_matrix.png", "图4-3 最优模型在测试集上的混淆矩阵", width_cm=11.5)

    # Main page 8: conclusion
    insert_before(body_anchor, "第五章 结论与展望", kind="h1", page_break_before=True)
    insert_before(body_anchor, "5.1 主要结论", kind="h2")
    insert_before(
        body_anchor,
        "第一，按题目规定的固定样本顺序完成了1300/299训练测试划分，并将特征选择、标准化、PCA和参数搜索严格限制在训练集内。"
        "第二，在多种候选模型中，树模型总体优于逻辑回归和多数类基线；经训练集5折调参后，SelectKBest+ExtraTrees获得"
        "0.4529的交叉验证macro-F1。第三，最终测试集accuracy为0.5518、MAE为0.5318，模型对主流质量等级5和6"
        "具有一定识别能力，但无法可靠区分稀有的极端等级。"
    )
    insert_before(body_anchor, "5.2 特征重要性与局限性", kind="h2")
    insert_before(
        body_anchor,
        "最终ExtraTrees模型的特征重要性显示，alcohol、volatile acidity和sulphates位列前三。"
        "该结果与单变量F检验并不完全相同，说明树模型还利用了变量之间的组合关系。需要注意的是，基于不纯度的特征重要性"
        "主要用于模型解释，不等价于因果结论。"
    )
    add_figure(body_anchor, "07_final_feature_importance.png", "图5-1 最终模型的特征重要性", width_cm=11.8)
    insert_before(
        body_anchor,
        "本研究的局限主要有三点：其一，固定顺序切分可能使训练集和测试集的分布存在差异；其二，质量等级严重不均衡，"
        "导致少数类的估计方差较大；其三，理化指标不能覆盖葡萄品种、品牌、价格和储存条件等信息。后续可在保证独立测试集"
        "不泄漏的前提下，尝试有序分类、代价敏感学习、重采样或校准概率，并收集更多高低质量等级样本。"
    )

    # Appendix page
    insert_before(body_anchor, "附录A 关键源代码", kind="h1", page_break_before=True)
    insert_before(
        body_anchor,
        "以下代码为Notebook中的关键实现片段，完整可运行代码及其原始输出保存在B_red_wine_quality_prediction.ipynb中。",
    )
    code_lines = [
        "df = pd.read_csv(DATA_PATH, sep=';')",
        "train_df = df.iloc[:1300].copy()",
        "test_df = df.iloc[1300:1599].copy()",
        "X_train, y_train = train_df[FEATURE_NAMES], train_df['quality']",
        "X_test, y_test = test_df[FEATURE_NAMES], test_df['quality']",
        "",
        "pipeline = Pipeline([",
        "    ('select', SelectKBest(score_func=f_classif)),",
        "    ('clf', ExtraTreesClassifier(random_state=42, n_jobs=1)),",
        "])",
        "param_grid = {",
        "    'select__k': [8, 10, 11],",
        "    'clf__n_estimators': [200],",
        "    'clf__max_depth': [None, 16],",
        "    'clf__min_samples_leaf': [2, 4],",
        "    'clf__max_features': ['sqrt', 0.8],",
        "    'clf__class_weight': [None, 'balanced'],",
        "}",
        "search = GridSearchCV(pipeline, param_grid, scoring=SCORING,",
        "                      refit='macro_f1', cv=CV, n_jobs=1)",
        "search.fit(X_train, y_train)",
        "best_model = search.best_estimator_",
        "y_pred = best_model.predict(X_test)",
        "proba = best_model.predict_proba(X_test)",
        "assert np.array_equal(y_pred, classes[np.argmax(proba, axis=1)])",
    ]
    for code_line in code_lines:
        insert_before(body_anchor, code_line if code_line else " ", kind="code")
    insert_before(
        body_anchor,
        "代码采用n_jobs=1以保证本地执行稳定，并在最终预测阶段同时保留predict与predict_proba + argmax的一致性核验。"
    )

    # Reference page stays in template section 3.
    # Index 136 carries the section break to the feedback sheet.  Clearing its
    # runs is safe and removes the template's reference-format instruction.
    for index in range(124, 139):
        clear_paragraph(source[index])
        source[index].style = document.styles["Normal"]
        reference_ppr = source[index]._p.get_or_add_pPr()
        reference_numbering = reference_ppr.find(qn("w:numPr"))
        if reference_numbering is not None:
            reference_ppr.remove(reference_numbering)
    style_heading(source[124], "参考文献", level=1)
    references = [
        "[1] Cortez P, Cerdeira A, Almeida F, Matos T, Reis J. Modeling wine preferences by data mining from physicochemical properties[J]. Decision Support Systems, 2009, 47(4): 547-553.",
        "[2] Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine learning in Python[J]. Journal of Machine Learning Research, 2011, 12: 2825-2830.",
        "[3] Geurts P, Ernst D, Wehenkel L. Extremely randomized trees[J]. Machine Learning, 2006, 63(1): 3-42.",
        "[4] Breiman L. Random forests[J]. Machine Learning, 2001, 45(1): 5-32.",
        "[5] Bishop C M. Pattern Recognition and Machine Learning[M]. New York: Springer, 2006.",
        "[6] Scikit-learn Developers. Scikit-learn User Guide[EB/OL]. https://scikit-learn.org/stable/user_guide.html, 2026-09-15.",
    ]
    for paragraph, reference in zip(source[125:131], references):
        write_paragraph(
            paragraph,
            reference,
            size=9,
            align=WD_ALIGN_PARAGRAPH.JUSTIFY,
            font_name=SERIF_FONT,
            first_indent=False,
            line_spacing=1.3,
            before=0,
            after=5,
        )
        paragraph.paragraph_format.left_indent = Cm(0.74)
        paragraph.paragraph_format.first_line_indent = Cm(-0.74)

    # Teacher feedback table remains blank, but its labels are restored in a readable font.
    set_cell_text(feedback_table.cell(0, 0), "任课教师评语", bold=True, size=11, shade=LIGHT_GRAY)
    set_cell_text(feedback_table.cell(1, 0), "成绩", bold=True, size=11, shade=LIGHT_GRAY)
    set_cell_text(feedback_table.cell(2, 0), "备注", bold=True, size=11, shade=LIGHT_GRAY)
    for row in feedback_table.rows:
        for cell in row.cells:
            set_cell_margins(cell, top=120, start=120, bottom=120, end=120)
            set_cell_border(cell, color="808080", size="8")
    for row, height in zip(feedback_table.rows, (Cm(11.5), Cm(1.0), Cm(2.5))):
        row.height = height
        row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
    feedback_table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Metadata hygiene
    props = document.core_properties
    props.title = "基于机器学习的红酒质量预测研究"
    props.subject = "机器学习课程大作业"
    props.author = ""
    props.comments = ""
    props.keywords = "红酒质量预测, 机器学习, ExtraTrees"
    document.save(OUTPUT_DOCX)
    print(f"created: {OUTPUT_DOCX}")


if __name__ == "__main__":
    main()
