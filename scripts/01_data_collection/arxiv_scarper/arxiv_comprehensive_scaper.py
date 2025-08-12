import arxiv
import pandas as pd
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_remaining_papers():
    """获取剩余的论文"""
    
    # 加载已有数据
    df_existing = pd.read_csv("human_abstracts.csv")
    existing_ids = set(df_existing['id'].tolist())
    all_papers = df_existing.to_dict('records')
    
    logger.info(f"已有 {len(all_papers)} 篇论文，还需要 {2500 - len(all_papers)} 篇")
    
    # 额外的查询策略
    queries = [
        # 机器学习相关
        "cat:cs.LG AND abs:image",
        "cat:cs.LG AND abs:visual",
        "cat:cs.LG AND abs:vision",
        
        # 人工智能相关
        "cat:cs.AI AND abs:perception",
        "cat:cs.AI AND abs:recognition",
        
        # 特定技术
        'abs:"object detection"',
        'abs:"image segmentation"',
        'abs:"facial recognition"',
        'abs:"image classification"',
        'abs:"computer vision"',
        
        # 数据集相关
        'abs:"ImageNet"',
        'abs:"COCO dataset"',
        'abs:"PASCAL VOC"',
        
        # 最新技术
        'abs:"vision transformer"',
        'abs:"DALL-E"',
        'abs:"diffusion model" AND abs:image',
        
        # 应用领域
        'abs:"medical imaging" AND cat:cs.CV',
        'abs:"autonomous driving" AND abs:vision',
        'abs:"surveillance" AND cat:cs.CV',
        
        # 不同年份的top论文
        'cat:cs.CV AND abs:"state-of-the-art"',
        'cat:cs.CV AND abs:"benchmark"',
        'cat:cs.CV AND abs:"survey"'
    ]
    
    client = arxiv.Client()
    
    for query in queries:
        if len(all_papers) >= 2500:
            break
            
        logger.info(f"\n查询: {query}")
        
        search = arxiv.Search(
            query=query,
            max_results=100,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        try:
            results = client.results(search)
            new_count = 0
            
            for result in results:
                if result.entry_id not in existing_ids:
                    paper_info = {
                        "id": result.entry_id,
                        "title": result.title.replace('\n', ' ').strip(),
                        "abstract": result.summary.replace('\n', ' ').strip()
                    }
                    all_papers.append(paper_info)
                    existing_ids.add(result.entry_id)
                    new_count += 1
                    
                    if len(all_papers) >= 2500:
                        break
            
            logger.info(f"新增 {new_count} 篇（总计: {len(all_papers)}）")
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"查询出错: {e}")
            continue
    
    # 保存结果
    df_final = pd.DataFrame(all_papers)
    df_final.to_csv("human_abstracts.csv", index=False, encoding='utf-8')
    
    logger.info(f"\n完成！总计 {len(all_papers)} 篇论文")
    
    if len(all_papers) >= 2500:
        logger.info("已达到目标 2500 篇！")
    else:
        logger.info(f"还差 {2500 - len(all_papers)} 篇")

if __name__ == "__main__":
    get_remaining_papers()