
try:
    embeddings = open('../data/code-100-formal.txt', 'r', encoding='UTF-8')
    code = open('../data/code-100.txt', 'r', encoding='UTF-8')
    embeddings = embeddings.readlines()
    code = code.readlines()

    for [embedding, code_line] in zip(embeddings, code):
        embedding = embedding.replace('\n', '')
        code_line = code_line.replace('\n', '')
        embedding = embedding[1:len(embedding) - 1]

        code_line = code_line.split(' ')
        embedding = embedding.split(', ')
        print(len(embedding))
        print(len(code_line))


except IOError:
    print("path not found")
