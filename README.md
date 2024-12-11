# 踩坑指北

## 显存问题
- 必须使用flash attn

## 安装版本
 ```
 transformers==4.46.3 #python 要3.9+
 qwen-vl-utils
 torch #根据需要安装
 flash-attn #根据版本下载whl安装，不要用pip直接在线安装
 ```

## 使用说明
### config 文件
```
path_7b: /home/pubw/proj/Qwen2-VL-7B-Instruct
path_2b: /home/pubw/proj/Qwen2-VL-2B-Instruct
json_path: /home/pubw/datasets/www25/test1/test1.json
image_path: /home/pubw/datasets/www25/test1/images
save_path: save
id: base
```
- json文件中的image 的path是图片名，不包括路径，方便起见单独配置image_path；
- 使用多个LLM，暂时使用2b和7b版本

### 添加自定义组合LLM的推理模型
- 继承InferModel_base 重写forward 代码实现复杂逻辑，[示例代码](model/infer_cot.py)
- 添加build文件参数 [代码](builder/inferModel_builder.py)
```
if config['model']=='cot':
    return infer_cot(config['path_2b'],config['path_7b'])
```
 - 输入是未经处理的json item 处理为qwen的输入后用LLM推理,调用
 ```
 self.infer_llm([messages1,messages2])


 messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
 ```

 ### 输出文件
```
/save/{id}/{timestamp}/output.jsonl
```
 