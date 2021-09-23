from api import app


if __name__ == '__main__':
    # print(analyse_texts("ты дебила кусок"))
    app.run(port=8000, debug=True)