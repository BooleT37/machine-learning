def combined_content(filenames):
    for name in filenames:
        yield open(name).read()