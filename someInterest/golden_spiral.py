"""
黄金螺旋线

- 黄金比例的定义是把一条线段分割为两部分，较短部分与较长部分长度之比等于较长部分与整体长度之比，其比值的近似值是0.618。
- 而斐波那契相邻两位数刚好符合这种黄金比例，而且前一项与后一项的比值无限接近0.618。
- 黄金螺线的画法：
    在以斐波那契数为边的正方形拼成的长方形中画一个90度的扇形，连起来的弧线就是斐波那契螺旋线。
    它来源于斐波那契数列（FibonacciSequence），又称黄金螺旋分割。
"""
import numpy as np
import turtle
import random

def generate_fibonacci(n):
    """
    fibonacci生成
    """
    if n == 0:
        return [1]
    elif n == 1:
        return [1]

    fib_list = [1, 1]

    for i in range(2, n):
        fib_list.append(fib_list[i-2]+fib_list[i-1])

    return fib_list

def draw(n):
    # turtle设置
    turtle.speed(10)
    # 绘制图形时笔的宽度
    turtle.pensize(5)
    # 放大倍率，用于更好的显示图形
    f0 = 50
    turtle.color("black")
    # 提笔
    turtle.penup()
    # 设置当前画笔位置为原点，朝向东
    turtle.home()
    # 落笔
    turtle.pendown()

    # 生成fibonacci数列
    fib_list = generate_fibonacci(n)

    # 遍历fibonacci数列，绘制黄金螺线
    for i in range(len(fib_list)):
        turtle.speed(1)
        turtle.pendown()
        # 画矩形
        if i == 0:
            fill_color = "black"
        else:
            fill_color = (random.random(), random.random(), random.random())
        # 绘制图形的填充颜色
        turtle.fillcolor(fill_color)
        # 准备开始填充图形
        turtle.begin_fill()
        # 画笔向绘制方向的当前方向移动distance(integer or float)的pixels距离
        turtle.forward(fib_list[i] * f0)
        # 逆时针移动90度
        turtle.left(90)
        turtle.forward(fib_list[i] * f0)
        turtle.left(90)
        turtle.forward(fib_list[i] * f0)
        turtle.left(90)
        turtle.forward(fib_list[i] * f0)
        turtle.left(90)
        # 填充完成
        turtle.end_fill()

        # 画圆弧
        # 随机产生填充颜色
        fill_color = (random.random(), random.random(), random.random())
        turtle.fillcolor(fill_color)
        if i == 0:
            # 画圆360度
            turtle.forward(fib_list[i] * f0 / 2)
            turtle.begin_fill()
            turtle.circle(fib_list[i] * f0 / 2, 360)
            turtle.end_fill()
            turtle.forward(fib_list[i] * f0 / 2)
            continue
        else:
            # 画圆弧90度
            turtle.begin_fill()
            turtle.circle(fib_list[i] * f0, 90)
            turtle.left(90)
            turtle.forward(fib_list[i] * f0)
            turtle.left(90)
            turtle.forward(fib_list[i] * f0)
            turtle.end_fill()

        # 移动到一下起点
        turtle.speed(0)
        turtle.penup()
        turtle.left(90)
        turtle.forward(fib_list[i] * f0)
        turtle.left(90)
        turtle.forward(fib_list[i] * f0)

    # 启动事件循环，turtle的最后一条语句
    turtle.done()

if __name__ == "__main__":
    n = 2
    draw(n)



