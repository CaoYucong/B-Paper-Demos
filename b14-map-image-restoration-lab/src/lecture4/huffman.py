"""Huffman coding for a finite alphabet. Decoding uses the same tree."""

from __future__ import annotations

import heapq
from dataclasses import dataclass


@dataclass
class HuffmanNode:
    frequency: float
    symbol: str | None = None
    left: HuffmanNode | None = None
    right: HuffmanNode | None = None


def frequencies(sequence: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in sequence:
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def build_tree(counts: dict[str, int]) -> HuffmanNode:
    if not counts:
        raise ValueError("Huffman coding needs at least one symbol.")
    heap: list[tuple[float, int, HuffmanNode]] = []
    for index, (symbol, count) in enumerate(sorted(counts.items())):
        heap.append((float(count), index, HuffmanNode(float(count), symbol)))
    heapq.heapify(heap)
    counter = len(heap)
    if len(heap) == 1:
        frequency, _index, node = heap[0]
        return HuffmanNode(frequency, None, node, HuffmanNode(0.0, None))
    while len(heap) > 1:
        freq_a, _index_a, left = heapq.heappop(heap)
        freq_b, _index_b, right = heapq.heappop(heap)
        parent = HuffmanNode(freq_a + freq_b, None, left, right)
        heapq.heappush(heap, (parent.frequency, counter, parent))
        counter += 1
    return heap[0][2]


def codes_from_tree(root: HuffmanNode) -> dict[str, str]:
    codes: dict[str, str] = {}

    def walk(node: HuffmanNode | None, prefix: str) -> None:
        if node is None:
            return
        if node.symbol is not None:
            codes[node.symbol] = prefix or "0"
            return
        walk(node.left, prefix + "0")
        walk(node.right, prefix + "1")

    walk(root, "")
    return codes


def encode(sequence: str, codes: dict[str, str]) -> str:
    return "".join(codes[symbol] for symbol in sequence)


def decode(bits: str, root: HuffmanNode) -> str:
    if not bits and root.symbol is not None:
        return ""
    output: list[str] = []
    node = root
    for bit in bits:
        if node is None or node.symbol is not None:
            raise ValueError("bit string left the tree.")
        node = node.left if bit == "0" else node.right
        if node is None:
            raise ValueError("bit string left the tree.")
        if node.symbol is not None:
            output.append(node.symbol)
            node = root
    if node is not root:
        raise ValueError("bit string ended inside a code.")
    return "".join(output)


def average_length(counts: dict[str, int], codes: dict[str, str]) -> float:
    total = sum(counts.values())
    return sum(counts[symbol] * len(codes[symbol]) for symbol in counts) / total


def tree_lines(root: HuffmanNode) -> list[str]:
    lines: list[str] = []

    def walk(node: HuffmanNode | None, prefix: str) -> None:
        if node is None:
            return
        if node.symbol is not None:
            shown = "space" if node.symbol == " " else node.symbol
            lines.append(f"{prefix or '0'}  {shown}  ({int(node.frequency)})")
            return
        walk(node.left, prefix + "0")
        walk(node.right, prefix + "1")

    walk(root, "")
    return lines
