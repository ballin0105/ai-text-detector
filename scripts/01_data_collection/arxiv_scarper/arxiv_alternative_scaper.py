import arxiv
import pandas as pd
import time
import logging
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_papers_by_search_terms():
    """
    使用不同的搜索词来获取论文，避开单一查询的限制
    """
    # 定义多个相关的搜索词
    search_terms = [
        "cat:cs.CV",
        "cat:cs.CV AND abs:deep",
        "cat:cs.CV AND abs:neural",
        "cat:cs.CV AND abs:image",
        "cat:cs.CV AND abs:detection",
        "cat:cs.CV AND abs:segmentation",
        "cat:cs.CV AND abs:recognition",
        "cat:cs.CV AND abs:classification",
        "cat:cs.CV AND abs:vision",
        "cat:cs.CV AND abs:visual",
        "cat:cs.CV AND abs:convolutional",
        "cat:cs.CV AND abs:transformer",
        "cat:cs.CV AND ti:deep",
        "cat:cs.CV AND ti:neural",
        "cat:cs.CV AND ti:vision"
    ]
    
    all_papers = []
    all_ids = set()
    
    for term in search_terms:
        logger.info(f"\n查询: {term}")
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=term,
            max_results=200,  # 每个查询最多200篇
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        try:
            results = client.results(search)
            count = 0
            
            for result in results:
                paper_id = result.entry_id
                
                # 去重
                if paper_id not in all_ids:
                    paper_info = {
                        "id": paper_id,
                        "title": result.title.replace('\n', ' ').strip(),
                        "abstract": result.summary.replace('\n', ' ').strip()
                    }
                    all_papers.append(paper_info)
                    all_ids.add(paper_id)
                    count += 1
            
            logger.info(f"获取 {count} 篇新论文（总计: {len(all_papers)}）")
            
            # 如果已经够了就停止
            if len(all_papers) >= 2500:
                break
                
            # 延迟避免请求过快
            time.sleep(3)
            
        except Exception as e:
            logger.error(f"查询出错: {e}")
            continue
    
    return all_papers

def get_papers_from_authors():
    """
    从知名作者获取论文
    """
    # CV领域的一些知名作者
    authors = [
        "Yann LeCun",
        "Geoffrey Hinton",
        "Andrew Zisserman",
        "Fei-Fei Li",
        "Kaiming He",
        "Ross Girshick",
        "Jitendra Malik",
        "Trevor Darrell"
    ]
    
    papers = []
    ids = set()
    
    for author in authors:
        query = f'au:"{author}" AND cat:cs.CV'
        logger.info(f"查询作者: {author}")
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=100,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        try:
            results = client.results(search)
            for result in results:
                if result.entry_id not in ids:
                    paper_info = {
                        "id": result.entry_id,
                        "title": result.title.replace('\n', ' ').strip(),
                        "abstract": result.summary.replace('\n', ' ').strip()
                    }
                    papers.append(paper_info)
                    ids.add(result.entry_id)
                    
        except Exception as e:
            logger.error(f"查询作者 {author} 时出错: {e}")
        
        time.sleep(2)
    
    return papers

def main():
    output_file = "human_abstracts.csv"
    
    print("arXiv论文抓取工具（替代方法）")
    print("-" * 50)
    
    # 加载已有数据
    existing_papers = []
    existing_ids = set()
    
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        existing_papers = df_existing.to_dict('records')
        existing_ids = set(df_existing['id'].tolist())
        print(f"已有 {len(existing_papers)} 篇论文")
    
    print("\n选择抓取策略：")
    print("1. 使用多个搜索词组合")
    print("2. 从知名作者获取论文")
    print("3. 两种方法都试试")
    
    choice = input("\n请选择 (1-3): ").strip()
    
    new_papers = []
    
    if choice in ["1", "3"]:
        print("\n使用搜索词组合方法...")
        term_papers = get_papers_by_search_terms()
        new_papers.extend(term_papers)
    
    if choice in ["2", "3"]:
        print("\n从知名作者获取...")
        author_papers = get_papers_from_authors()
        new_papers.extend(author_papers)
    
    # 合并和去重
    added_count = 0
    for paper in new_papers:
        if paper['id'] not in existing_ids:
            existing_papers.append(paper)
            existing_ids.add(paper['id'])
            added_count += 1
    
    # 保存结果
    if added_count > 0:
        df = pd.DataFrame(existing_papers)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n成功添加 {added_count} 篇新论文")
        print(f"总计: {len(existing_papers)} 篇")
    
    if len(existing_papers) < 2500:
        print(f"\n还需要 {2500 - len(existing_papers)} 篇论文")

if __name__ == "__main__":
    main()