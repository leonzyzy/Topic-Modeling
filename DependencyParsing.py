import re

# Basic Tokenizer
def tokenize(text):
    return re.findall(r'\b\w+\b', text)

# Simple POS Tagger using a dictionary (for demonstration purposes)
def pos_tag(tokens):
    pos_dict = {
        "The": "DET", "quick": "ADJ", "brown": "ADJ", "fox": "NOUN", 
        "jumps": "VERB", "over": "PREP", "the": "DET", "lazy": "ADJ", "dog": "NOUN",
        "A": "DET", "fast": "ADJ", "dark-colored": "ADJ", "leaps": "VERB", "sleepy": "ADJ",
        "a": "DET", "man": "NOUN", "with": "PREP", "hat": "NOUN", "is": "VERB",
        "walking": "VERB", "in": "PREP", "park": "NOUN"
    }
    return [(token, pos_dict.get(token, "UNK")) for token in tokens]

# Tokenize and POS tag
def process_sentence(sentence):
    tokens = tokenize(sentence)
    pos_tags = pos_tag(tokens)
    return tokens, pos_tags

# Basic Dependency Parser (using simple rules)
def dependency_parse(pos_tags):
    deps = []
    stack = []
    
    for i, (token, pos) in enumerate(pos_tags):
        if pos in ["NOUN", "PRON"]:
            if stack and stack[-1][1] in ["DET", "ADJ"]:
                deps.append((stack.pop()[0], "mod", token))
        if pos == "VERB":
            if stack and stack[-1][1] in ["NOUN", "PRON"]:
                deps.append((stack.pop()[0], "subj", token))
            if i < len(pos_tags) - 1 and pos_tags[i+1][1] in ["PREP"]:
                deps.append((token, "prep", pos_tags[i+1][0]))
        if pos == "PREP":
            if i < len(pos_tags) - 1 and pos_tags[i+1][1] in ["DET", "NOUN"]:
                deps.append((token, "pobj", pos_tags[i+1][0]))
        stack.append((token, pos))
    
    return deps

# Tree Node
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f"TreeNode({self.value})"

def build_tree(tokens, dependencies):
    token_to_node = {token: TreeNode(token) for token in tokens}
    root = None
    for head, dep, child in dependencies:
        head_node = token_to_node[head]
        child_node = token_to_node[child]
        head_node.add_child(child_node)
        if dep == "subj":
            root = head_node
    return root or token_to_node[tokens[0]]

def print_tree(node, level=0):
    print('  ' * level + repr(node))
    for child in node.children:
        print_tree(child, level + 1)

# Process a sample sentence
sentence = "A man with a hat is walking in the park."
tokens, pos_tags = process_sentence(sentence)
dependencies = dependency_parse(pos_tags)
print("Dependencies:")
for dep in dependencies:
    print(f"{dep[0]} -> {dep[1]} -> {dep[2]}")

# Build and print the dependency tree
tree = build_tree(tokens, dependencies)
print("\nDependency Tree:")
print_tree(tree)
