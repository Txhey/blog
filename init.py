import os
import glob
import json
import time


def find_cover_image(img_folder, file_name):
    # 构建cover图片文件名
    cover_image_name = f"cover-{file_name}"
    # 搜索img文件夹中名为cover-{file_name}的图片文件
    img_files = glob.glob(os.path.join(img_folder, f'{cover_image_name}.*'))
    if img_files:
        # 返回第一个找到的cover图片文件的文件名（包括后缀）
        return os.path.basename(img_files[0])
    else:
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
        creation_time = os.path.getctime(md_file)
        modification_time = os.path.getmtime(md_file)

        # 假设存在同名的.tconf文件
        tconf_file = md_file.replace('.md', '.tconf')
        taglist = []
        grouplist = []

        if os.path.exists(tconf_file):
            with open(tconf_file, 'r', encoding='utf-8') as f:
                tconf_data = json.load(f)
                taglist = tconf_data.get('tagList', [])
                grouplist = tconf_data.get('groupList', [])
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
            img_url = f"https://raw.githubusercontent.com/Txhey/note/main/main/{relative_path}/img/{cover_image}"
        else:
            img_url = None

        # 将文件信息添加到字典中
        file_info_dict[file_key] = {
            'relative_path': relative_path,
            'file_name': file_name,  # 包含.md后缀
            'creation_time': creation_time,
            'modification_time': modification_time,
            'taglist': taglist,
            'grouplist': grouplist,
            'img': img_url
        }

    # 根据创建时间对file_info_dict的key进行排序
    sorted_keys = sorted(file_info_dict.keys(), key=lambda k: file_info_dict[k]['creation_time'])

    # 创建排序后的列表，用于sortByCreateTime字段
    sortByCreateTime = [file_key for file_key in sorted_keys]

    # 创建总的结构
    structure = {
        "fileList": file_info_dict,
        "tagSummary": tag_summary,
        "groupSummary": group_summary,
        "sortByCreateTime": sortByCreateTime
    }

    # 将结构保存到structure.json文件中
    with open('structure.json', 'w', encoding='utf-8') as f:
        json.dump(structure, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # 设定main目录的路径
    base_path = os.path.join(os.path.dirname(__file__), '.', 'main')
    search_md_files(base_path)
