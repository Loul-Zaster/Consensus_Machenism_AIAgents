with open('agent.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix indentation after try statements
content = content.replace('try:\n        self.vectorstore', 'try:\n            self.vectorstore')
content = content.replace('try:\n        self.initial_ingest', 'try:\n            self.initial_ingest')
content = content.replace('try:\n        for filename', 'try:\n            for filename')
content = content.replace('try:\n        retrieved_docs', 'try:\n            retrieved_docs')
content = content.replace('try:\n        filename =', 'try:\n            filename =')

# Write the fixed content back to the file
with open('agent.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Indentation fixed successfully!')
