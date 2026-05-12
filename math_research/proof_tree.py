import json
from typing import Any, Dict, List, Optional

class ProofTree:
    """管理数学证明逻辑结构的单例类"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProofTree, cls).__new__(cls)
            cls._instance.nodes = {}  # id -> {"description": str, "status": str, "dependencies": list}
        return cls._instance

    def reset(self):
        self.nodes = {}

    def _save_state(self):
        try:
            with open("proof_tree_state.json", "w", encoding="utf-8") as f:
                json.dump({"proof_tree": self.nodes}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def add_node(self, node_id: str, description: str, dependencies: List[str] = None) -> Dict[str, Any]:
        if dependencies is None:
            dependencies = []
            
        # 检查循环依赖
        for dep in dependencies:
            if dep not in self.nodes:
                return {"error": f"依赖的节点 {dep} 不存在，请先注册该节点。"}
                
        self.nodes[node_id] = {
            "description": description,
            "status": "Unproven",
            "dependencies": dependencies
        }
        self._save_state()
        return {"success": True, "message": f"节点 {node_id} 注册成功。"}

    def update_status(self, node_id: str, status: str, justification: str) -> Dict[str, Any]:
        if node_id not in self.nodes:
            return {"error": f"节点 {node_id} 不存在。"}
        if status not in ["Unproven", "Proved", "Refuted"]:
            return {"error": "状态必须是 Unproven, Proved 或 Refuted"}
            
        self.nodes[node_id]["status"] = status
        self.nodes[node_id]["last_justification"] = justification
        self._save_state()
        return {"success": True, "message": f"节点 {node_id} 状态已更新为 {status}。"}

    def get_tree_state(self) -> Dict[str, Any]:
        return {"proof_tree": self.nodes}

def proof_tree_manager(action: str, **kwargs) -> Dict[str, Any]:
    """
    管理数学证明的逻辑结构树。
    极其宽容的参数解析，支持 AI 传入各种别名。
    """
    tree = ProofTree()
    
    # 尝试从 kwargs 或嵌套的 dict 中提取 node_id
    _node_id = kwargs.get("node_id") or kwargs.get("lemma_id") or kwargs.get("id") or kwargs.get("name")
    
    # 尝试提取描述
    _description = kwargs.get("description") or kwargs.get("statement") or kwargs.get("root_statement") or kwargs.get("content")
    
    # 提取依赖
    dependencies = kwargs.get("dependencies", [])
    if isinstance(dependencies, str):
        dependencies = [dependencies]
        
    status = kwargs.get("status")
    justification = kwargs.get("justification") or kwargs.get("proof") or kwargs.get("reason")
    
    # 支持 init_tree 这种一次性建树的操作
    if action == "init_tree":
        tree.reset()
        if _description:
            tree.add_node("Main", _description, [])
        lemmas = kwargs.get("lemmas", [])
        if isinstance(lemmas, list):
            for l in lemmas:
                l_id = l.get("id") or l.get("lemma_id")
                l_desc = l.get("statement") or l.get("description")
                l_deps = l.get("dependencies", [])
                if l_id and l_desc:
                    tree.add_node(l_id, l_desc, l_deps)
        return tree.get_tree_state()
    
    if action in ["add_node", "add_lemma"]:
        if not _node_id or not _description:
            return {"error": f"缺少节点ID或描述。你传入的参数是: {kwargs}"}
        return tree.add_node(_node_id, _description, dependencies)
        
    elif action == "update_status":
        if not _node_id or not status or not justification:
            return {"error": f"更新状态需要 ID, status 和 理由。你传入的是: {kwargs}"}
        # 兼容状态别名
        if status.lower() in ["proved", "proven", "true", "yes"]:
            status = "Proved"
        elif status.lower() in ["refuted", "false", "no", "failed"]:
            status = "Refuted"
        else:
            status = "Unproven"
        return tree.update_status(_node_id, status, justification)
        
    elif action in ["get_state", "verify_tree"]:
        return tree.get_tree_state()
        
    elif action == "reset":
        tree.reset()
        return {"success": True, "message": "证明树已重置。"}
        
    return {"error": f"未知操作: {action}。可用操作: add_node, update_status, get_state, init_tree"}