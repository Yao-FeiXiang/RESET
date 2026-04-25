# README

## 文件结构

```
large-scale-preprocess/
├── │   py/                     # 预处理入口
├── │   cpp/                    # 预处理单元
├── │   java/                   # 前置:将WebGraph代码处理为BCSR格式
├── │   gsh-2015/               # 数据集
│   ├── Config.java
│   ├── CsrWriter.java
│   ├── GraphUtils.java
│   └── Main.java
├── pom.xml
├── README.md
└── run_java.sh
```

## 相关说明
WebGraph 并没有提供java之外的处理接口
所以预处理需要先将原.graph处理为其他语言可用识别的类型 