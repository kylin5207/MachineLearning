"""
正多边形绘制
内角和+外角和=180
正多边形外角和=360，每一个定点的外角=360/n
每个内角+每个外角=180，可以得到正多边形的每个内角=180(n-2)/n，因此正多边形的内角和=180(n-2)
"""
import turtle

# numinput是turtle提供的供用户输入浮点型数字的地方
# 注意：通过numinput录入的数据是浮点型数据
n = int(turtle.numinput("输入正多边形的边数","边数"))

def drawShape(color1):
    # 为turtle设置方向
    turtle.seth(0)                   # 0表示东，90北，180西，270南
    turtle.pencolor(color1)          # 画笔颜色

    step = (turtle.window_width())/(2*n)

    # 绘制正多边形的每一条边
    for m in range(1,n+1):
         turtle.pendown()            # 画笔落下，留下痕迹
         turtle.forward(step)          # 向前移动distance

         turtle.right(360/n)         #外角    # 相对角度
         # turtle.right(180-180*(n-2)/n)#外角=180-内角


if n <= 2:
    print("抱歉，您输入的边数错误！！！")
else:
    turtle.setup(1024, 600, 100, 100)
    # 绘画窗口的宽度、高度，左上角的X坐标、Y坐标。
    turtle.speed(2)  # 速度
    turtle.pencolor("white")
    turtle.goto(-50, 50)  # 从点【-50,50】开始
    drawShape("red")
    turtle.done()
