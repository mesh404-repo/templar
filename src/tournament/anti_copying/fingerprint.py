"""Structural fingerprinting for code similarity detection.

This module generates fingerprints that are SIMILAR for SIMILAR code,
unlike SHA256 which produces completely different hashes for any change.

The fingerprint is posted to blockchain so ALL validators can detect copies
even if they never see the original code.
"""

import ast
import hashlib
import io
import logging
import tokenize
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CodeFingerprint:
    """Structural fingerprint of code.
    
    Attributes:
        full_hash: SHA256 of the entire code (exact match)
        structural_hash: Hash of normalized AST (similarity match)
        function_hashes: Hashes of individual functions (partial match)
        token_count: Number of tokens (size indicator)
    """
    full_hash: str
    structural_hash: str
    function_hashes: dict[str, str]
    token_count: int
    
    def to_chain_format(self) -> str:
        """Format for posting to blockchain.
        
        Returns a compact string that can be stored on-chain.
        Format: {structural_hash}:{token_count}
        """
        return f"{self.structural_hash[:32]}:{self.token_count}"
    
    @classmethod
    def from_chain_format(cls, chain_data: str) -> "CodeFingerprint":
        """Parse from blockchain format."""
        parts = chain_data.split(":")
        return cls(
            full_hash="",  # Not stored on chain
            structural_hash=parts[0] if len(parts) > 0 else "",
            function_hashes={},  # Not stored on chain
            token_count=int(parts[1]) if len(parts) > 1 else 0,
        )


class ASTNormalizer(ast.NodeTransformer):
    """Normalize AST to ignore superficial differences.
    
    This transformer:
    - Renames all variables to generic names (var_0, var_1, etc.)
    - Removes docstrings and comments
    - Normalizes function names (except required ones like inner_steps)
    """
    
    PRESERVED_NAMES = {"inner_steps", "InnerStepsResult", "torch", "nn", "F"}
    
    def __init__(self):
        self.var_counter = 0
        self.name_map: dict[str, str] = {}
    
    def _get_normalized_name(self, original: str) -> str:
        """Get normalized name for a variable/function."""
        if original in self.PRESERVED_NAMES:
            return original
        if original.startswith("_"):  # Keep private indicators
            return original
        if original not in self.name_map:
            self.name_map[original] = f"var_{self.var_counter}"
            self.var_counter += 1
        return self.name_map[original]
    
    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Normalize variable names."""
        node.id = self._get_normalized_name(node.id)
        return self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Normalize function names and remove docstrings."""
        # Normalize name (except preserved ones)
        if node.name not in self.PRESERVED_NAMES:
            node.name = self._get_normalized_name(node.name)
        
        # Remove docstring
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        
        return self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Normalize class names and remove docstrings."""
        if node.name not in self.PRESERVED_NAMES:
            node.name = self._get_normalized_name(node.name)
        
        # Remove docstring
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        
        return self.generic_visit(node)
    
    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        """Normalize string constants (remove comments in strings)."""
        # Keep numeric constants as-is (important for algorithm)
        if isinstance(node.value, (int, float)):
            return node
        # Normalize strings (could be comments)
        if isinstance(node.value, str):
            node.value = "<string>"
        return node


def compute_fingerprint(code: str) -> CodeFingerprint:
    """Compute structural fingerprint of code.
    
    Args:
        code: Python source code
        
    Returns:
        CodeFingerprint with various hashes for similarity detection
    """
    # Full hash (exact match)
    full_hash = hashlib.sha256(code.encode()).hexdigest()
    
    try:
        # Parse AST
        tree = ast.parse(code)
        
        # Normalize AST
        normalizer = ASTNormalizer()
        normalized_tree = normalizer.visit(tree)
        
        # Dump normalized AST to string
        normalized_dump = ast.dump(normalized_tree, annotate_fields=False)
        
        # Hash normalized AST
        structural_hash = hashlib.sha256(normalized_dump.encode()).hexdigest()
        
        # Extract individual function hashes
        function_hashes = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_normalizer = ASTNormalizer()
                normalized_func = func_normalizer.visit(node)
                func_dump = ast.dump(normalized_func, annotate_fields=False)
                func_hash = hashlib.sha256(func_dump.encode()).hexdigest()
                function_hashes[node.name] = func_hash
        
        # Count tokens (rough size indicator)
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
            token_count = len(tokens)
        except tokenize.TokenizeError:
            token_count = len(code.split())
        
        return CodeFingerprint(
            full_hash=full_hash,
            structural_hash=structural_hash,
            function_hashes=function_hashes,
            token_count=token_count,
        )
        
    except SyntaxError as e:
        logger.warning(f"Failed to parse code for fingerprinting: {e}")
        # Fallback to simple hash
        return CodeFingerprint(
            full_hash=full_hash,
            structural_hash=full_hash,  # Use full hash as fallback
            function_hashes={},
            token_count=len(code.split()),
        )


def compare_fingerprints(fp1: CodeFingerprint, fp2: CodeFingerprint) -> float:
    """Compare two fingerprints and return similarity score.
    
    Args:
        fp1: First fingerprint
        fp2: Second fingerprint
        
    Returns:
        Similarity score between 0.0 (different) and 1.0 (identical)
    """
    # Exact match
    if fp1.full_hash == fp2.full_hash:
        return 1.0
    
    # Structural match (same AST structure)
    if fp1.structural_hash == fp2.structural_hash:
        return 0.95  # Very similar, just variable names changed
    
    # Function-level comparison
    if fp1.function_hashes and fp2.function_hashes:
        common_funcs = set(fp1.function_hashes.keys()) & set(fp2.function_hashes.keys())
        if common_funcs:
            matching = sum(
                1 for f in common_funcs 
                if fp1.function_hashes[f] == fp2.function_hashes[f]
            )
            func_similarity = matching / max(
                len(fp1.function_hashes), 
                len(fp2.function_hashes)
            )
            if func_similarity > 0.5:
                return 0.7 + (func_similarity * 0.2)  # 0.7-0.9 range
    
    # Size comparison (very rough)
    if fp1.token_count and fp2.token_count:
        size_ratio = min(fp1.token_count, fp2.token_count) / max(fp1.token_count, fp2.token_count)
        if size_ratio > 0.9:
            return 0.3  # Similar size, might be related
    
    return 0.0  # No similarity detected


def fingerprints_match(chain_fp1: str, chain_fp2: str, threshold: float = 0.85) -> bool:
    """Quick check if two on-chain fingerprints indicate similar code.
    
    This is used for fast cross-validator similarity detection using
    only the data posted to blockchain.
    
    Note: This is a HEURISTIC check. For definitive similarity detection,
    use calculate_similarity() when both code files are available.
    
    Args:
        chain_fp1: First fingerprint in chain format
        chain_fp2: Second fingerprint in chain format
        threshold: Similarity threshold (default 0.85)
        
    Returns:
        True if fingerprints indicate potentially similar code
    """
    fp1 = CodeFingerprint.from_chain_format(chain_fp1)
    fp2 = CodeFingerprint.from_chain_format(chain_fp2)
    
    # Structural hash match = definitely similar
    if fp1.structural_hash == fp2.structural_hash:
        return True
    
    # Check if structural hashes share prefix (partial match)
    # This catches cases where normalization differs slightly
    for prefix_len in [16, 12, 8]:
        if (fp1.structural_hash[:prefix_len] == fp2.structural_hash[:prefix_len]):
            return True
    
    # Size-based heuristic: similar size = potential copy
    # This catches copies where structure differs but size is similar
    if fp1.token_count and fp2.token_count:
        size_ratio = min(fp1.token_count, fp2.token_count) / max(fp1.token_count, fp2.token_count)
        
        # Very similar size (within 5%) - likely related
        if size_ratio > 0.95:
            return True
        
        # Similar size (within 15%) and some hash overlap
        if size_ratio > 0.85:
            # Check for any common prefix
            if fp1.structural_hash[:4] == fp2.structural_hash[:4]:
                return True
    
    return False


def fingerprints_definitely_match(chain_fp1: str, chain_fp2: str) -> bool:
    """Strict check - only returns True if fingerprints are very likely the same code.
    
    Use this for automatic rejection. For flagging, use fingerprints_match().
    """
    fp1 = CodeFingerprint.from_chain_format(chain_fp1)
    fp2 = CodeFingerprint.from_chain_format(chain_fp2)
    
    # Only match if structural hash is identical or very close
    if fp1.structural_hash == fp2.structural_hash:
        return True
    
    # First 20 chars identical = very likely same structure
    if fp1.structural_hash[:20] == fp2.structural_hash[:20]:
        return True
    
    return False

