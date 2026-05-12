import os
import sys
import json
import time
import requests

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gene_research.gene_database import GeneDatabase, GeneRecord

def build_full_genome_db(db_path: str):
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = GeneDatabase(db_path)
    print(f"==================================================")
    print(f" 开始构建全量离线人类基因组数据库 (The 20k Base Dump)")
    print(f" 目标文件: {db_path}")
    print(f"==================================================")
    
    # UniProt REST API: search for all human reviewed proteins (Swiss-Prot)
    # This usually yields around 20,400 entries.
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    query = "reviewed:true AND organism_id:9606"
    batch_size = 500
    
    url = f"{base_url}?query={query}&format=json&size={batch_size}"
    
    total_fetched = 0
    total_inserted = 0
    
    while url:
        print(f"[{time.strftime('%H:%M:%S')}] 正在请求批量数据... (已获取: {total_fetched})")
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 429:
                print("触发 API 速率限制，休眠 10 秒...")
                time.sleep(10)
                continue
            response.raise_for_status()
        except Exception as e:
            print(f"[错误] 请求失败: {e}。等待重试...")
            time.sleep(5)
            continue
            
        data = response.json()
        results = data.get("results", [])
        if not results:
            break
            
        genes_batch = []
        for protein in results:
            try:
                genes_data = protein.get("genes", [])
                symbol = ""
                if genes_data:
                    symbol = genes_data[0].get("geneName", {}).get("value", "")
                
                accession = protein.get("primaryAccession", "")
                if not symbol: 
                    symbol = accession
                
                name = protein.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "Unknown Protein")
                
                # 提取 GO terms
                go_terms = []
                for db_ref in protein.get("uniProtKBCrossReferences", []):
                    if db_ref.get("database") == "GO":
                        go_id = db_ref.get("id")
                        go_name = ""
                        for prop in db_ref.get("properties", []):
                            if prop.get("key") == "GoTerm":
                                go_name = prop.get("value", "")
                        go_terms.append({"name": go_name, "namespace": go_id})
                        
                gene_record = GeneRecord(
                    gene_id=accession,
                    symbol=symbol.upper(),
                    name=name[:100], # 截断超长名称
                    aliases="",
                    chromosome="",
                    start_pos=0, end_pos=0,
                    strand="+",
                    gene_type="protein_coding",
                    summary=f"Full dump from UniProt Swiss-Prot.",
                    go_terms_json=json.dumps(go_terms[:10], ensure_ascii=False), # 只存前10个避免过大
                    pathways_json="[]", 
                    domains_json="[]", 
                    diseases_json="[]",
                    sequence_nt="",
                    sequence_aa=""
                )
                genes_batch.append(gene_record)
            except Exception as e:
                print(f"解析记录报错: {e}")
                continue
                
        if genes_batch:
            db.insert_genes(genes_batch)
            total_inserted += len(genes_batch)
            
        total_fetched += len(results)
        print(f" -> 成功插入 {len(genes_batch)} 条记录，当前数据库总量: {total_inserted}")
        
        # 获取下一页的链接
        links = response.headers.get("Link")
        url = None
        if links:
            for link in links.split(","):
                if 'rel="next"' in link:
                    url = link[link.find("<")+1 : link.find(">")]
                    break
                    
    db.connection.commit()
    db.close()
    print(f"\n==================================================")
    print(f" 全量数据库构建完成！")
    print(f" 最终写入基因数量: {total_inserted}")
    print(f" 数据库文件位置: {os.path.abspath(db_path)}")
    print(f"==================================================")

if __name__ == "__main__":
    build_full_genome_db("gene_knowledge_full_20k.db")
