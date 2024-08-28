import trafilatura

def handle_predict(content):
    main_content = trafilatura.extract(content)
    if len(main_content) > 100:
        return 1
    else:
        return 0
    
def handle_extract(content):
    main_content = trafilatura.extract(content)
    return main_content