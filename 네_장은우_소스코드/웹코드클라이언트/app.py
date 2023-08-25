import collections
from flask import Flask, render_template, request
import os
import random
import os


app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
##    print(per1)4\
    per = 0.5
    return render_template("info.html", per = per)


@app.route('/voca', methods=['GET', 'POST'])
def voca():
    vlad=[]
    f = os.listdir('D:\최종본 웹코드\static')

    count='※'
    for i in f:
        if "wav" in i:
            print(i)
            vlad.append(i)
    if len(vlad)==8:
        pass
    else:
        while 1:
            vlad.append(count)
            if len(vlad)==8:
                break
    print(vlad)

    return render_template("yock1.html", aa = "거북이", bb="북이 옛날에 토끼와 거북이", cc="았습니다 야 토끼", dd="야 토끼야", ee=vlad[4], ff=vlad[5], gg=vlad[6], hh=vlad[7])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
##    app.run(debug= False)
