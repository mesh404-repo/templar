"""Code similarity detection using multiple methods.

This module provides detailed similarity analysis between two code files,
used for local validation when both codes are available.
"""

import ast
import difflib
import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class CodeSimilarity:
    """Detailed similarity analysis between two code files."""
    
    # Overall similarity (0.0 to 1.0)
    overall_score: float
    
    # Individual method scores
    text_similarity: float      # Raw text diff ratio
    ast_similarity: float       # AST structure similarity
    function_similarity: float  # Function-by-function comparison
    
    # Verdict
    is_copy: bool
    confidence: Literal["high", "medium", "low"]
    reason: str


def _get_text_similarity(code1: str, code2: str) -> float:
    """Calculate text-based similarity using difflib."""
    return difflib.SequenceMatcher(None, code1, code2).ratio()


def _normalize_ast(tree: ast.AST) -> str:
    """Normalize AST by removing names and dumping structure."""
    
    class Normalizer(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = "VAR"
            return self.generic_visit(node)
        
        def visit_FunctionDef(self, node):
            if node.name != "inner_steps":
                node.name = "FUNC"
            # Remove docstring
            if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant)):
                node.body = node.body[1:]
            return self.generic_visit(node)
        
        def visit_arg(self, node):
            node.arg = "ARG"
            return self.generic_visit(node)
    
    normalized = Normalizer().visit(tree)
    return ast.dump(normalized, annotate_fields=False)


def _get_ast_similarity(code1: str, code2: str) -> float:
    """Calculate AST structure similarity."""
    try:
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        
        norm1 = _normalize_ast(tree1)
        norm2 = _normalize_ast(tree2)
        
        return difflib.SequenceMatcher(None, norm1, norm2).ratio()
    except SyntaxError:
        return 0.0


def _extract_functions(code: str) -> dict[str, str]:
    """Extract function bodies as normalized strings."""
    try:
        tree = ast.parse(code)
        functions = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get function body as string
                func_body = ast.dump(node, annotate_fields=False)
                functions[node.name] = func_body
        
        return functions
    except SyntaxError:
        return {}


def _get_function_similarity(code1: str, code2: str) -> float:
    """Compare functions between two code files."""
    funcs1 = _extract_functions(code1)
    funcs2 = _extract_functions(code2)
    
    if not funcs1 or not funcs2:
        return 0.0
    
    # Compare inner_steps specifically (most important)
    if "inner_steps" in funcs1 and "inner_steps" in funcs2:
        inner_sim = difflib.SequenceMatcher(
            None, funcs1["inner_steps"], funcs2["inner_steps"]
        ).ratio()
        
        # inner_steps is weighted heavily
        return inner_sim * 0.8 + 0.2 * _compare_other_functions(funcs1, funcs2)
    
    return _compare_other_functions(funcs1, funcs2)


def _compare_other_functions(funcs1: dict, funcs2: dict) -> float:
    """Compare non-inner_steps functions."""
    # Remove inner_steps for this comparison
    f1 = {k: v for k, v in funcs1.items() if k != "inner_steps"}
    f2 = {k: v for k, v in funcs2.items() if k != "inner_steps"}
    
    if not f1 or not f2:
        return 0.0
    
    # Find best matches
    total_sim = 0.0
    matched = 0
    
    for name1, body1 in f1.items():
        best_match = 0.0
        for name2, body2 in f2.items():
            sim = difflib.SequenceMatcher(None, body1, body2).ratio()
            best_match = max(best_match, sim)
        if best_match > 0.5:
            total_sim += best_match
            matched += 1
    
    if matched == 0:
        return 0.0
    
    return total_sim / max(len(f1), len(f2))


def calculate_similarity(code1: str, code2: str) -> CodeSimilarity:
    """Calculate detailed similarity between two code files.
    
    Args:
        code1: First code file
        code2: Second code file
        
    Returns:
        CodeSimilarity with detailed analysis
    """
    # Calculate individual similarities
    text_sim = _get_text_similarity(code1, code2)
    ast_sim = _get_ast_similarity(code1, code2)
    func_sim = _get_function_similarity(code1, code2)
    
    # Weighted overall score
    # AST and function similarity are more important than text
    overall = (
        text_sim * 0.2 +
        ast_sim * 0.4 +
        func_sim * 0.4
    )
    
    # Determine if it's a copy
    is_copy = False
    confidence: Literal["high", "medium", "low"] = "low"
    reason = "No significant similarity detected"
    
    if overall > 0.95:
        is_copy = True
        confidence = "high"
        reason = "Nearly identical code (>95% match)"
    elif overall > 0.85:
        is_copy = True
        confidence = "high"
        reason = f"Very similar code structure ({overall:.0%} match)"
    elif overall > 0.70:
        is_copy = True
        confidence = "medium"
        reason = f"Significant similarity detected ({overall:.0%} match)"
    elif ast_sim > 0.85:
        is_copy = True
        confidence = "medium"
        reason = f"Same code structure with different variable names (AST: {ast_sim:.0%})"
    elif func_sim > 0.85:
        is_copy = True
        confidence = "medium"
        reason = f"Same function implementations ({func_sim:.0%} function match)"
    elif overall > 0.50:
        is_copy = False
        confidence = "low"
        reason = f"Some similarity but likely coincidental ({overall:.0%})"
    
    return CodeSimilarity(
        overall_score=overall,
        text_similarity=text_sim,
        ast_similarity=ast_sim,
        function_similarity=func_sim,
        is_copy=is_copy,
        confidence=confidence,
        reason=reason,
    )

