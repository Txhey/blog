import os
import glob
import json
import time
import re
from datetime import datetime


# 获取文件摘要（去除标题符号、代码块、表格、空白符）
def get_title_and_extract_from_md(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # 获取title
        title = "title not found"
        for line in f:
            # 使用正则表达式匹配以#开头的行，并捕获标题内容
            match = re.match(r'^\s*#\s*(.+)', line)
            if match:
                title = match.group(1)
                break
        # print(title)
        # 读取 Markdown 文件内容
        html = f.read()
        # 去掉列表
        html = re.sub(r'\* (.*?)[\r\n]', r' \1 ', html)
        # 标题去除#
        html = re.sub(r'#* (.*?)[\r\n]', r"\1", html)
        # 去除表格
        html = re.sub(r'\|.*?\|.*?\n', "", html)
        # 去除代码块
        html = re.sub(r'```.*?```|~~~.*?~~~', "", html, flags=re.DOTALL)
        # 去掉图片
        html = re.sub(r'!\[.*?\]\(.*?\)', " ", html, flags=re.DOTALL)
        # 去除换行、空行、制表符
        html = re.sub(r'\s+', " ", html)
        return title, html.strip()[:200]


def find_cover_image(img_folder, file_key):
    # 尝试找cover-{file_key}命名的图片
    cover_image_name = f"cover-{file_key}"
    img_files = glob.glob(os.path.join(img_folder, f'{cover_image_name}.*'))
    if img_files:
        return os.path.basename(img_files[0])

    # 找不到cover-{file_key}命名的图片时，尝试找img文件夹中以cover开头的任意图片
    img_files = glob.glob(os.path.join(img_folder, 'cover*'))
    if img_files:
        return os.path.basename(img_files[0])

    # 找不到cover-{file_key}命名的图片时，尝试找img文件夹中以cover开头的任意图片
    img_files = glob.glob(os.path.join(img_folder, '*'))
    if img_files:
        return os.path.basename(img_files[0])

    return None


def search_md_files(base_path):
    # 搜索main目录及其子目录下的.md文件，忽略以~开头的文件夹
    md_files = []
    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if not d.startswith('~') and not d.startswith('.')]
        md_files.extend(glob.glob(os.path.join(root, '*.md')))

    file_info_dict = {}
    tag_summary = {}
    group_summary = {}

    for md_file in md_files:
        # 获取文件的相对路径
        relative_path = os.path.relpath(os.path.dirname(md_file), base_path)
        # 获取文件名（包含后缀）
        file_name = os.path.basename(md_file)
        # 获取文件名（不包含后缀）
        file_key = os.path.splitext(file_name)[0]
        # 获取文件创建时间和最后修改时间
        creation_time = time.ctime(os.path.getctime(md_file))
        modification_time = time.ctime(os.path.getmtime(md_file))
        # 获取文件的摘要
        title, abstract = get_title_and_extract_from_md(os.path.join(base_path, md_file))

        # 假设存在同名的.tconf文件
        tconf_file = md_file.replace('.md', '.tconf')
        taglist = []
        grouplist = []

        if os.path.exists(tconf_file):
            with open(tconf_file, 'r', encoding='utf-8') as f:
                print(tconf_file)
                tconf_data = json.load(f)
                tagList = tconf_data.get('tagList', [])
                groupList = tconf_data.get('groupList', [])
        else:
            # 如果.tconf文件不存在，则创建一个带有默认内容的.json文件
            default_tconf_content = {"tagList": [], "groupList": []}
            with open(tconf_file, 'w', encoding='utf-8') as f:
                json.dump(default_tconf_content, f, ensure_ascii=False)

        # 更新tag_summary和group_summary
        for tag in taglist:
            if tag not in tag_summary:
                tag_summary[tag] = []
            tag_summary[tag].append(file_key)

        for group in grouplist:
            if group not in group_summary:
                group_summary[group] = []
            group_summary[group].append(file_key)

        # 搜索img文件夹中名为cover-{file_key}的图片文件
        img_folder = os.path.join(base_path, relative_path, 'img')
        cover_image = find_cover_image(img_folder, file_key)

        if cover_image:
            # 构建img字段的URL
            img_url = f"https://raw.githubusercontent.com/Txhey/blog/main/main/{relative_path}/img/{cover_image}"
        else:
            img_url = None

        # 将文件信息添加到字典中
        file_info_dict[file_key] = {
            "key": file_key,
            'relative_path': relative_path,
            'file_name': file_name,  # 包含.md后缀
            'title': title,
            'abstract': abstract,
            'creation_time': creation_time,
            'modification_time': modification_time,
            'tagList': tagList,
            'groupList': groupList,
            'img': img_url
        }

    # 根据创建时间对file_info_dict的key进行排序
    file_list = list(file_info_dict.values())

    # 按照创建时间排序
    sortByCreateTimeList = sorted(file_list,
                                  key=lambda x: datetime.strptime(x['creation_time'], "%a %b %d %H:%M:%S %Y"),
                                  reverse=True)
    # 按modification_time排序
    sortByLastModifyList = sorted(file_list,
                              key=lambda x: datetime.strptime(x['modification_time'], "%a %b %d %H:%M:%S %Y"),
                              reverse=True)

    # 创建排序后的列表，用于sortByCreateTime字段
    sortByCreateTime = [item["key"] for item in sortByCreateTimeList]
    sortByLastModify = [item["key"] for item in sortByLastModifyList]
    # 创建总的结构
    structure = {
        "fileList": file_info_dict,
        "tagSummary": tag_summary,
        "groupSummary": group_summary,
        "sortByCreateTime": sortByCreateTime,
        "sortByLastModify": sortByLastModify
    }

    # 将结构保存到structure.json文件中
    with open('structure.json', 'w', encoding='utf-8') as f:
        json.dump(structure, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # 设定main目录的路径
    base_path = os.path.join(os.path.dirname(__file__), '.', 'main')
    search_md_files(base_path)
