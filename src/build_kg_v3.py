#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task B: PieStream KG 构建 v3.0
适配实际数据格式（7588 wikisource判例 + 9440 Excel企业数据）
规则优先策略：从结构化字段直接映射，避免LLM调用成本

产出:
  data/kg/case_graphs/case_*.json       — 每个案例一个图谱
  data/kg/enterprise_graph.json          — 企业全景图
  data/kg/enterprise_subgraphs/COMP_*.json — 企业3-hop子图
  data/kg/README.md                      — 统计报告
"""

import json, os, sys, re, hashlib
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

# ============================================================
# 路径配置
# ============================================================
BASE = Path("C:/workspace/paper")
RAW_CASES = BASE / "data" / "raw" / "cases"
RAW_ENTERPRISES = BASE / "data" / "raw" / "enterprises"
KG_DIR = BASE / "data" / "kg"
CASE_GRAPHS_DIR = KG_DIR / "case_graphs"
ENT_SUBGRAPHS_DIR = KG_DIR / "enterprise_subgraphs"
ENT_GRAPH_PATH = KG_DIR / "enterprise_graph.json"
README_PATH = KG_DIR / "README.md"
PROGRESS_PATH = KG_DIR / "TASK_B_PROGRESS.md"

for d in [CASE_GRAPHS_DIR, ENT_SUBGRAPHS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 工具函数
# ============================================================
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def make_id(prefix, text):
    """生成稳定的短ID"""
    h = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    return f"{prefix}_{h}"

def normalize_name(name):
    """标准化公司名/人名"""
    if not name:
        return ""
    name = name.strip()
    # 移除常见后缀用于匹配
    for suffix in ['有限责任公司', '股份有限公司', '有限公司', '普通合伙', '有限合伙']:
        name = name.replace(suffix, '')
    return name.strip()

def is_chinese(text):
    """检查是否包含中文"""
    return bool(re.search(r'[\u4e00-\u9fff]', text)) if text else False

# ============================================================
# Step 0: 加载企业索引（用于匹配 Location/Industry）
# ============================================================
def load_enterprise_index():
    """加载企业数据，建立 name->info 和 id->info 索引"""
    log("加载企业索引...")
    ent_by_id = {}
    ent_by_name_short = {}
    
    ent_files = sorted(RAW_ENTERPRISES.glob("comp_*.json"))
    for ef in ent_files:
        try:
            d = json.load(open(ef, 'r', encoding='utf-8'))
            cid = d.get('company_id', '')
            name = d.get('name', '')
            ent_by_id[cid] = d
            # 短名索引
            short = normalize_name(name)
            if short and len(short) >= 4:
                if short not in ent_by_name_short:
                    ent_by_name_short[short] = []
                ent_by_name_short[short].append(d)
        except:
            pass
    
    log(f"  加载了 {len(ent_by_id)} 企业 (by_id), {len(ent_by_name_short)} 企业 (by_name)")
    return ent_by_id, ent_by_name_short

def match_enterprise(company_name, ent_by_name_short):
    """用公司名模糊匹配企业数据，返回最佳匹配"""
    if not company_name:
        return None
    short = normalize_name(company_name)
    if not short or len(short) < 4:
        return None
    
    # 精确匹配短名
    if short in ent_by_name_short:
        candidates = ent_by_name_short[short]
        if len(candidates) == 1:
            return candidates[0]
        # 多个候选，用完整名匹配
        for c in candidates:
            if c['name'] == company_name:
                return c
        return candidates[0]  # 返回第一个
    
    # 子串匹配
    best = None
    best_len = 0
    for sname, candidates in ent_by_name_short.items():
        if len(sname) >= 6 and (sname in short or short in sname):
            if len(sname) > best_len:
                best_len = len(sname)
                best = candidates[0]
    return best

# ============================================================
# Step 1: 案例 -> 案例图谱
# ============================================================
def extract_case_graph(case_data, ent_by_name_short):
    """
    从单个案例提取图谱
    返回: case_graph dict (符合TASK_B.md格式)
    """
    case_id = case_data.get('case_id', '')
    case_type = case_data.get('case_type', '其他')
    court = case_data.get('court', '')
    decision_date = case_data.get('decision_date', '')
    defendants = case_data.get('defendants', [])
    key_facts = case_data.get('key_facts', {})
    narratives = case_data.get('narratives', [])
    
    nodes = []
    edges = []
    
    # ---- 实体抽取 ----
    # 1. Company 实体 (从 defendants)
    companies_in_case = []
    for d in defendants:
        if d.get('type') == 'company':
            name = d['name']
            cid = make_id("COMP", f"{case_id}:{name}")
            ent_match = match_enterprise(name, ent_by_name_short)
            
            props = {
                "case_id": case_id,
                "source": case_data.get('source', ''),
                "role": d.get('role', '')
            }
            if ent_match:
                props["matched_enterprise_id"] = ent_match["company_id"]
                props["industry"] = ent_match.get("industry_name", "")
                props["province"] = ent_match.get("registered_province", "")
                props["city"] = ent_match.get("registered_city", "")
            
            nodes.append({
                "id": cid,
                "type": "Company",
                "label": name,
                "properties": props
            })
            companies_in_case.append({"id": cid, "name": name, "match": ent_match})
    
    # 2. Person 实体 (从 defendants)
    persons_in_case = []
    for d in defendants:
        if d.get('type') == 'person':
            name = d['name']
            pid = make_id("PERSON", f"{case_id}:{name}")
            nodes.append({
                "id": pid,
                "type": "Person",
                "label": name,
                "properties": {
                    "case_id": case_id,
                    "role": d.get('role', ''),
                    "source": case_data.get('source', '')
                }
            })
            persons_in_case.append({"id": pid, "name": name, "role": d.get('role', '')})
    
    # 3. Authority 实体
    auth_name = court if court else "税务机关"
    aid = make_id("AUTH", auth_name)
    nodes.append({
        "id": aid,
        "type": "Authority",
        "label": auth_name,
        "properties": {}
    })
    
    # 4. ViolationEvent 实体
    scheme_desc = key_facts.get('scheme_description', '')
    violation_item = case_type
    for n in defendants:
        if n.get('type') == 'company':
            violation_item = f"{case_type}: {n['name']}" if case_type else n['name']
            break
    if not scheme_desc:
        scheme_desc = violation_item
    
    veid = make_id("EVENT", f"{case_id}")
    nodes.append({
        "id": veid,
        "type": "ViolationEvent",
        "label": scheme_desc[:200],
        "properties": {
            "violation_item": case_type,
            "scheme_description": scheme_desc,
            "case_type": case_type,
            "legal_basis": case_data.get('case_number', '')
        }
    })
    
    # 5. FinancialFlow 实体 (从 key_facts 金额)
    amount_involved = key_facts.get('amount_involved', 0)
    tax_evaded = key_facts.get('tax_evaded', 0)
    
    if amount_involved:
        fid1 = make_id("FLOW", f"{case_id}:amount_involved")
        nodes.append({
            "id": fid1,
            "type": "FinancialFlow",
            "label": f"涉案金额: {amount_involved}元",
            "properties": {
                "value": amount_involved,
                "type": "amount_involved",
                "unit": "元",
                "case_id": case_id
            }
        })
    else:
        fid1 = None
    
    if tax_evaded and tax_evaded != amount_involved:
        fid2 = make_id("FLOW", f"{case_id}:tax_evaded")
        nodes.append({
            "id": fid2,
            "type": "FinancialFlow",
            "label": f"逃税金额: {tax_evaded}元",
            "properties": {
                "value": tax_evaded,
                "type": "tax_evaded",
                "unit": "元",
                "case_id": case_id
            }
        })
    else:
        fid2 = None
    
    # 6. Location 和 Industry (从企业匹配)
    for comp in companies_in_case:
        if comp['match']:
            ent = comp['match']
            # Location
            prov = ent.get('registered_province', '')
            city = ent.get('registered_city', '')
            if prov and prov != '未知':
                loc_id = make_id("LOC", prov)
                existing_loc = any(n['id'] == loc_id for n in nodes)
                if not existing_loc:
                    nodes.append({
                        "id": loc_id,
                        "type": "Location",
                        "label": prov,
                        "properties": {"level": "province"}
                    })
                edges.append({
                    "source": comp['id'],
                    "target": loc_id,
                    "relation": "located_in",
                    "properties": {"case_id": case_id}
                })
            # Industry
            industry = ent.get('industry_name', '')
            if industry and industry != '未知':
                ind_id = make_id("IND", industry)
                existing_ind = any(n['id'] == ind_id for n in nodes)
                if not existing_ind:
                    nodes.append({
                        "id": ind_id,
                        "type": "Industry",
                        "label": industry,
                        "properties": {}
                    })
                edges.append({
                    "source": comp['id'],
                    "target": ind_id,
                    "relation": "in_industry",
                    "properties": {"case_id": case_id}
                })
    
    # ---- 关系抽取 ----
    # penalized_by: Company -> Authority
    for comp in companies_in_case:
        edges.append({
            "source": comp['id'],
            "target": aid,
            "relation": "penalized_by",
            "properties": {"case_id": case_id, "date": decision_date}
        })
    
    # committed: Company -> ViolationEvent
    for comp in companies_in_case:
        edges.append({
            "source": comp['id'],
            "target": veid,
            "relation": "committed",
            "properties": {"case_id": case_id, "date": decision_date}
        })
    
    # controlled_by: Company -> Person
    for person in persons_in_case:
        for comp in companies_in_case:
            edges.append({
                "source": comp['id'],
                "target": person['id'],
                "relation": "controlled_by",
                "properties": {"case_id": case_id, "role": person['role']}
            })
    
    # involves_amount: ViolationEvent -> FinancialFlow
    if fid1:
        edges.append({
            "source": veid,
            "target": fid1,
            "relation": "involves_amount",
            "properties": {"case_id": case_id, "type": "amount_involved"}
        })
    if fid2:
        edges.append({
            "source": veid,
            "target": fid2,
            "relation": "involves_amount",
            "properties": {"case_id": case_id, "type": "tax_evaded"}
        })
    
    # co_penalized: Company <-> Company (同案企业)
    for i in range(len(companies_in_case)):
        for j in range(i+1, len(companies_in_case)):
            edges.append({
                "source": companies_in_case[i]['id'],
                "target": companies_in_case[j]['id'],
                "relation": "co_penalized",
                "properties": {"case_id": case_id}
            })
    
    # ---- 因果链标注 ----
    causal_chain = build_causal_chain(case_type, case_id, defendants, key_facts, 
                                       companies_in_case, persons_in_case)
    
    # ---- 构建输出 ----
    case_graph = {
        "case_id": case_id,
        "case_type": case_type,
        "source": case_data.get("source", ""),
        "graph": {
            "nodes": nodes,
            "edges": edges
        },
        "causal_chain": causal_chain,
        "metrics": {
            "tax_evaded": tax_evaded,
            "amount_involved": amount_involved,
            "node_count": len(nodes),
            "edge_count": len(edges)
        }
    }
    
    return case_graph

# ============================================================
# Step 2: 因果链构建
# ============================================================
CAUSAL_PATTERNS = {
    "虚开增值税专用发票": {
        "pattern": "票货分离",
        "flags": ["has_fake_invoice", "has_fund_flow_back", "has_fee_deduction"],
        "steps": [
            {"order": 1, "action": "虚构交易", "actor": "开票方", "target": "受票方"},
            {"order": 2, "action": "开具增值税专用发票", "actor": "开票方", "target": "受票方"},
            {"order": 3, "action": "对公付款", "actor": "受票方", "target": "开票方"},
            {"order": 4, "action": "资金回流（私户）", "actor": "开票方", "target": "实控人"},
            {"order": 5, "action": "申报抵扣税款", "actor": "受票方", "target": "税务机关"}
        ]
    },
    "隐匿收入": {
        "pattern": "私户收款",
        "flags": ["has_personal_account", "has_income_concealment", "has_two_set_books"],
        "steps": [
            {"order": 1, "action": "消费者付款至私户", "actor": "消费者", "target": "个人账户"},
            {"order": 2, "action": "资金汇集至财务人员", "actor": "个人账户", "target": "财务负责人"},
            {"order": 3, "action": "转入实控人私户", "actor": "财务负责人", "target": "实控人私户"},
            {"order": 4, "action": "对公账户仅申报合规收入", "actor": "公司", "target": "税务机关"},
            {"order": 5, "action": "私户资金用于个人消费", "actor": "实控人私户", "target": "个人消费"}
        ]
    },
    "偷税": {
        "pattern": "隐匿收入",
        "flags": ["has_income_concealment", "has_false_declaration"],
        "steps": [
            {"order": 1, "action": "隐匿销售收入", "actor": "纳税人", "target": "税务机关"},
            {"order": 2, "action": "虚假纳税申报", "actor": "纳税人", "target": "税务机关"},
            {"order": 3, "action": "少缴应纳税款", "actor": "纳税人", "target": "税务机关"},
            {"order": 4, "action": "税务稽查发现", "actor": "税务机关", "target": "纳税人"}
        ]
    },
    "骗取出口退税": {
        "pattern": "假报出口",
        "flags": ["has_export_fraud", "has_fake_customs", "has_false_declaration"],
        "steps": [
            {"order": 1, "action": "伪造报关单证", "actor": "出口企业", "target": "海关"},
            {"order": 2, "action": "虚报出口货物", "actor": "出口企业", "target": "税务机关"},
            {"order": 3, "action": "申请出口退税", "actor": "出口企业", "target": "税务机关"},
            {"order": 4, "action": "骗取退税款", "actor": "出口企业", "target": "国库"}
        ]
    },
    "虚开发票": {
        "pattern": "无真实交易开票",
        "flags": ["has_fake_invoice", "has_no_real_transaction"],
        "steps": [
            {"order": 1, "action": "虚构交易合同", "actor": "开票方", "target": "受票方"},
            {"order": 2, "action": "开具发票", "actor": "开票方", "target": "受票方"},
            {"order": 3, "action": "收取开票费", "actor": "受票方", "target": "开票方"},
            {"order": 4, "action": "受票方入账抵扣", "actor": "受票方", "target": "税务机关"}
        ]
    },
    "转让定价": {
        "pattern": "高进低出",
        "flags": ["has_cross_border", "has_transfer_pricing", "has_profit_shift"],
        "steps": [
            {"order": 1, "action": "高价向境外关联方采购", "actor": "境内公司", "target": "境外关联方"},
            {"order": 2, "action": "低价向境外关联方销售", "actor": "境内公司", "target": "境外关联方"},
            {"order": 3, "action": "利润转移至低税地", "actor": "境内公司", "target": "境外关联方"},
            {"order": 4, "action": "境内申报微利或亏损", "actor": "境内公司", "target": "税务机关"}
        ]
    }
}

def build_causal_chain(case_type, case_id, defendants, key_facts, companies, persons):
    """根据 case_type 生成因果链"""
    pattern = CAUSAL_PATTERNS.get(case_type)
    if not pattern:
        # 尝试模糊匹配
        for key in CAUSAL_PATTERNS:
            if key in case_type or case_type in key:
                pattern = CAUSAL_PATTERNS[key]
                break
    if not pattern:
        pattern = {
            "pattern": "通用违法",
            "flags": ["has_violation"],
            "steps": [
                {"order": 1, "action": "违法行为发生", "actor": "纳税人", "target": ""},
                {"order": 2, "action": "税务稽查", "actor": "税务机关", "target": "纳税人"},
                {"order": 3, "action": "行政处罚/刑事追究", "actor": "税务机关/法院", "target": "纳税人"}
            ]
        }
    
    # 用实际数据填充步骤中的占位符
    comp_names = [c['name'] for c in companies]
    person_names = [p['name'] for p in persons]
    
    filled_steps = []
    for step in pattern["steps"]:
        s = dict(step)
        # 简单替换
        if s['actor'] in ['开票方', '受票方', '出口企业', '境内公司', '纳税人', '公司']:
            if comp_names:
                s['actor'] = comp_names[0][:20] if len(comp_names) == 1 else f"{comp_names[0][:15]}等"
        if s['target'] in ['开票方', '受票方', '出口企业', '境外关联方', '纳税人', '公司']:
            if len(comp_names) > 1:
                s['target'] = comp_names[1][:20]
            elif comp_names:
                s['target'] = comp_names[0][:20]
        if '实控人' in s.get('actor', '') and person_names:
            s['actor'] = person_names[0]
        if '实控人' in s.get('target', '') and person_names:
            s['target'] = person_names[0]
        filled_steps.append(s)
    
    return {
        "pattern": pattern["pattern"],
        "flags": pattern["flags"],
        "case_type": case_type,
        "steps": filled_steps
    }

# ============================================================
# Step 3: 企业全景图谱（聚合）
# ============================================================
def build_enterprise_graph(all_case_graphs):
    """聚合所有案例图谱为统一企业全景图"""
    log("构建企业全景图谱...")
    
    # 收集所有唯一节点和边
    node_index = {}  # id -> node
    edge_index = set()  # (source, target, relation) -> edge
    
    for cg in all_case_graphs:
        g = cg.get("graph", {})
        for node in g.get("nodes", []):
            nid = node["id"]
            if nid not in node_index:
                node_index[nid] = node
            else:
                # 合并属性
                existing = node_index[nid]
                for k, v in node.get("properties", {}).items():
                    if k not in existing.get("properties", {}):
                        existing.setdefault("properties", {})[k] = v
        
        for edge in g.get("edges", []):
            key = (edge["source"], edge["target"], edge["relation"])
            edge_index.add((key, edge))
    
    # 转为列表
    nodes = list(node_index.values())
    edges = [e[1] for e in edge_index]
    
    # 统计
    node_types = Counter(n['type'] for n in nodes)
    edge_types = Counter(e['relation'] for e in edges)
    company_count = node_types.get('Company', 0)
    authority_count = node_types.get('Authority', 0)
    location_count = node_types.get('Location', 0)
    industry_count = node_types.get('Industry', 0)
    
    graph = {
        "meta": {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "node_types": list(node_types.keys()),
            "edge_types": list(edge_types.keys()),
            "node_type_distribution": dict(node_types),
            "edge_type_distribution": dict(edge_types),
            "company_count": company_count,
            "authority_count": authority_count,
            "location_count": location_count,
            "industry_count": industry_count,
            "cases_processed": len(all_case_graphs)
        },
        "nodes": nodes,
        "edges": edges
    }
    
    return graph

# ============================================================
# Step 4: 企业子图提取（3-hop BFS）
# ============================================================
def extract_all_subgraphs(enterprise_graph):
    """对所有 Company 节点提取 3-hop 子图"""
    log("提取企业子图 (3-hop BFS)...")
    
    # 构建邻接表
    adj_out = defaultdict(set)
    adj_in = defaultdict(set)
    node_types = {}
    
    for node in enterprise_graph["nodes"]:
        node_types[node["id"]] = node["type"]
    
    for edge in enterprise_graph["edges"]:
        s, t = edge["source"], edge["target"]
        adj_out[s].add(t)
        adj_in[t].add(s)
    
    # 只对 Company 类型提取子图
    company_nodes = [n for n in enterprise_graph["nodes"] if n["type"] == "Company"]
    log(f"  待处理企业: {len(company_nodes)}")
    
    # 预先建立 node_id -> full_node 的映射
    node_map = {n["id"]: n for n in enterprise_graph["nodes"]}
    edge_list = enterprise_graph["edges"]
    
    count = 0
    for cn in company_nodes:
        cid = cn["id"]
        sg = extract_khop_subgraph_bfs(cid, adj_out, adj_in, node_map, edge_list, k=3)
        
        if sg:
            out_path = ENT_SUBGRAPHS_DIR / f"{cid}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(sg, f, ensure_ascii=False, indent=2)
            count += 1
            
            if count % 100 == 0:
                log(f"  已提取 {count}/{len(company_nodes)} 个子图...")
    
    log(f"  子图提取完成: {count}/{len(company_nodes)}")
    return count

def extract_khop_subgraph_bfs(center_id, adj_out, adj_in, node_map, edge_list, k=3):
    """BFS k-hop 子图提取"""
    visited = {center_id}
    frontier = {center_id}
    
    for _ in range(k):
        new_frontier = set()
        for node in frontier:
            new_frontier.update(adj_out.get(node, set()))
            new_frontier.update(adj_in.get(node, set()))
        new_frontier -= visited
        visited.update(new_frontier)
        frontier = new_frontier
        if not frontier:
            break
    
    # 收集节点和边
    sub_nodes = [node_map[nid] for nid in visited if nid in node_map]
    visited_set = visited
    sub_edges = [e for e in edge_list 
                 if e["source"] in visited_set and e["target"] in visited_set]
    
    return {
        "center_node": center_id,
        "hop": k,
        "num_nodes": len(sub_nodes),
        "num_edges": len(sub_edges),
        "nodes": sub_nodes,
        "edges": sub_edges
    }

# ============================================================
# 主流程
# ============================================================
def main():
    log("=" * 60)
    log("Task B: PieStream KG 构建 v3.0")
    log("=" * 60)
    
    # Step 0: 加载企业索引
    ent_by_id, ent_by_name_short = load_enterprise_index()
    
    # Step 1+2: 处理所有案例
    log("\n----- 阶段1: 案例图谱生成 -----")
    case_files = sorted(RAW_CASES.glob("case_*.json"))
    # 排除汇总文件
    case_files = [cf for cf in case_files if not cf.name.startswith('_')]
    log(f"待处理案例: {len(case_files)}")
    
    all_case_graphs = []
    success = 0
    errors = 0
    batch_size = 500
    
    for i, cf in enumerate(case_files):
        try:
            with open(cf, 'r', encoding='utf-8') as f:
                case_data = json.load(f)
            
            case_graph = extract_case_graph(case_data, ent_by_name_short)
            
            # 保存个案图谱
            out_name = cf.name
            out_path = CASE_GRAPHS_DIR / out_name
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(case_graph, f, ensure_ascii=False, indent=2)
            
            all_case_graphs.append(case_graph)
            success += 1
            
            if (i + 1) % batch_size == 0:
                log(f"  已处理 {i+1}/{len(case_files)} ({100*(i+1)//len(case_files)}%)")
        except Exception as e:
            errors += 1
            if errors <= 5:
                log(f"  ERROR {cf.name}: {e}")
    
    log(f"\n案例图谱: {success} 成功, {errors} 失败")
    
    # Step 3: 企业全景图
    log("\n----- 阶段2: 企业全景图谱 -----")
    enterprise_graph = build_enterprise_graph(all_case_graphs)
    
    with open(ENT_GRAPH_PATH, 'w', encoding='utf-8') as f:
        json.dump(enterprise_graph, f, ensure_ascii=False, indent=2)
    
    meta = enterprise_graph["meta"]
    log(f"  节点: {meta['num_nodes']}, 边: {meta['num_edges']}")
    log(f"  节点类型: {dict(meta['node_type_distribution'])}")
    log(f"  边类型: {dict(meta['edge_type_distribution'])}")
    
    # Step 4: 企业子图
    log("\n----- 阶段3: 企业子图提取 -----")
    sub_count = extract_all_subgraphs(enterprise_graph)
    
    # Step 5: 统计报告
    log("\n----- 阶段4: 统计报告 -----")
    case_type_dist = Counter(cg['case_type'] for cg in all_case_graphs)
    total_nodes = sum(cg['metrics']['node_count'] for cg in all_case_graphs)
    total_edges = sum(cg['metrics']['edge_count'] for cg in all_case_graphs)
    avg_nodes = total_nodes / max(len(all_case_graphs), 1)
    avg_edges = total_edges / max(len(all_case_graphs), 1)
    
    readme = f"""# 知识图谱构建报告 (v3.0)

## 案例图谱
- 总数: {len(all_case_graphs)} 个
- 平均节点数: {avg_nodes:.1f}
- 平均边数: {avg_edges:.1f}
- 案例类型分布: {dict(case_type_dist)}

## 企业全景图谱
- 节点总数: {meta['num_nodes']}
- 边总数: {meta['num_edges']}
- 节点类型分布: {dict(meta['node_type_distribution'])}
- 边类型分布: {dict(meta['edge_type_distribution'])}
- 企业数量: {meta.get('company_count', 0)}
- 稽查局数量: {meta.get('authority_count', 0)}
- 地域节点: {meta.get('location_count', 0)}
- 行业节点: {meta.get('industry_count', 0)}

## 企业子图
- 子图总数: {sub_count}
- 提取方式: 3-hop BFS

## 构建信息
- 构建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 输入案例: {len(all_case_graphs)} 份
- 企业数据: {len(ent_by_id)} 家
"""
    
    with open(README_PATH, 'w', encoding='utf-8') as f:
        f.write(readme)
    
    # 进度日志
    progress = f"""# Task B 执行日志

## 完成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 产出清单
| 文件 | 说明 |
|------|------|
| `data/kg/case_graphs/case_*.json` | {len(all_case_graphs)} 个案例图谱 |
| `data/kg/enterprise_graph.json` | 企业全景图谱 ({meta['num_nodes']}节点/{meta['num_edges']}边) |
| `data/kg/enterprise_subgraphs/COMP_*.json` | {sub_count} 个企业子图 |
| `data/kg/README.md` | KG 统计报告 |
"""
    with open(PROGRESS_PATH, 'w', encoding='utf-8') as f:
        f.write(progress)
    
    log("\n" + "=" * 60)
    log("Task B 完成!")
    log(f"  案例图谱: {len(all_case_graphs)}")
    log(f"  企业全景图: {meta['num_nodes']} 节点, {meta['num_edges']} 边")
    log(f"  企业子图: {sub_count}")
    log("=" * 60)

if __name__ == "__main__":
    main()
