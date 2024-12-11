task2_label_list="""
[\"实物拍摄(含售后)\",\"商品分类选项\",\"商品头图\",\"商品详情页截图\",\"下单过程中出现异常（显示购买失败浮窗）\",\"订单详情页面\",\"支付页面\",\"消费者与客服聊天页面\",\"评论区截图页面\",\"物流页面-物流列表页面\",\"物流页面-物流跟踪页面\",\"物流页面-物流异常页面\",\"退款页面\",\"退货页面\",\"换货页面\",\"购物车页面\",\"店铺页面\",\"活动页面\",\"优惠券领取页面\",\"账单/账户页面\",\"个人信息页面\",\"投诉举报页面\",\"平台介入页面\",\"外部APP截图\",\"其他类别图片\"]
"""

PROMPTS={}
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|完成|>"
PROMPTS['DESC_CAPTION']="""[{0}]位于[{2}],属于图标种类是[{1}],它的具体描述是[{3}], 分析它属于[{label}]的原因是{4}"""


PROMPTS['structure_CAPTION']="""
目标：
给定输入图片获取图片内全部的全部图标，至少写10个。

对于每个识别的图标，提取以下信息：
- 图标名 具体的手机图标按钮名称或者实物名称
- 图标种类 图标种类包含[导航类图标、操作类图标、内容展示类图标、状态反馈类图标、交互类图标、提示与通知类图标、实物]
- 布局 图标之间的关系或者是页面内的位置信息
- 具体信息 图标具体什么样子，颜色大小与形状是什么，如果有字符，具体是什么


要求输出格式：
(图标名<|>图标种类<|>布局<|>具体信息<|>标签分析)##
(图标名<|>图标种类<|>布局<|>具体信息<|>标签分析)##
<|完成|>
######################
例子1 Output:
(退货运费险说明弹窗<|>提示与通知类图标<|>页面中间<|>白底黑字，内容为退货相关信息，带有一个橙色“我知道了”按钮<|>##
(我知道了按钮<|>操作类图标<|>页面底部<|>矩形橙色按钮，白色文字“我知道了”##
<|完成|>


例子2 Output:
(继续付款按钮<|>操作类图标<|>页面底部<|>矩形橙色按钮，白色文字“继续付款”##
(商品列表<|>内容展示类图标<|>页面中部<|>展示购买的商品信息，包括商品图片和价格##
(价格明细<|>内容展示类图标<|>页面中部<|>展示商品总价、优惠信息和最终支付金额##
(支付方式选择<|>交互类图标<|>页面底部<|>展示支付方式选项，如支付宝##
(订单信息<|>内容展示类图标<|>页面中部<|>展示订单编号、配送服务等信息##
<|完成|>

例子3 Output:
(会员信息<|>内容展示类图标<|>页面中部<|>展示会员信息，包括会员类型和价格##
(立即激活按钮<|>操作类图标<|>页面中部<|>矩形棕色按钮，白色文字“立即激活”##
(活动信息<|>内容展示类图标<|>页面中部<|>展示限时活动信息，包括活动名称和参与方式##
(积分信息<|>内容展示类图标<|>页面中部<|>展示积分信息，包括积分数量和积分使用方式##
<|完成|>
######################
-Real Data-
######################
我的输入图片是<image>
######################
Output:
"""

PROMPTS['get_attribute']="""
目标：
识别图片中的图标，提取信息

######################
例子1 Output:
1.生活卡选项，显示生活卡（原88VIP）的信息，并提供与其他卡种类的比较
2.优酷/芒果等图标，展示了会员服务（如优酷、芒果）的特权费用和内容说明
3.购物卡与全能卡对比表,通过详细数据和文字比较三种卡的权益（如退货、红包）
4.当前已选提示,显示“当前已选88VIP生活卡”，表示当前状态和用户选择

例子2 Output:
1.地板瓷砖，显示了部分损坏的木纹地板瓷砖，边缘有破损
2.白色墙面,展现了一面白色的墙面


例子3 Output:
1.直播按钮,显示“直播”功能，提供活动期间实时商品展示和互动
2.搜索框,用户可以搜索商品、优惠或活动信息
3.618狂欢节横幅,突出“618狂欢节低至1元”字样，强调活动主题和优惠力度
4.超级补贴区块,展示多种商品的优惠补贴价格，附有限时倒计时

例子4 Output:
1.悬浮窗， 悬浮窗显示购买失败
######################
-Real Data-
######################
我的输入图片是<image>
######################
Output:
"""
PROMPTS['attribute_caption']="""[{0}]位于[{2}],属于图标种类是[{1}],它的具体描述是[{3}], 分析它属于[{label}]的原因是{4}"""

PROMPTS['summary']="""
- 目标: 总结该类不同例子中的共性图标
- 输入: 输入不同例子的描述, 包含图标位置,图标种类,图标描述,分类分析
- 输出: 该类共有特性[图标位置,图标种类,图标共性描述,特性分析]

图标种类一共有：[标题,导航栏,工具栏,内容区,按钮,输入框,标签,底部导航栏,浮动操作按钮,状态栏,对话框,卡片,图片或图标,列表,拍摄实物]
要求输出格式：
(图标位置<|>图标种类<|>图标共性描述<|>特性分析)##
(图标位置<|>图标种类<|>图标共性描述<|>特性分析)##
(图标位置<|>图标种类<|>图标共性描述<|>特性分析)##
######################
例子1 Output:
(页面底部<|>底部导航栏<|>包含导航选项，如“首页”、“消息”、“购物车”等<|>提供页面快速切换功能，提高用户操作效率)##
(页面顶部<|>工具栏<|>购物车图标与数量展示<|>提供购物车管理功能，便于用户快速查看购物车内容)##
(页面中部<|>内容区<|>显示已选商品的详细信息，包括商品图片、名称、价格和数量<|>作为购物车页面的核心内容，提供商品展示及选择信息)##

例子2 Output:
(页面全部<|>拍摄实物<|>拍摄产品照片，属于实物拍摄<|>拍摄实物照片,非手机截图，属于实物拍摄(含售后))##
######################
-Real Data-
######################
{text}
######################
Output:

"""


PROMPTS['local_summary']="""目标:输入标签和各个样本信息，总结该标签的样本中最可能出现哪些图标，并分析原因
输入: 标签名称,区域,图标
输入图标格式：(图标名<|>图标种类<|>布局<|>具体信息<|>标签分析)
对于具体每个图标：
- 图标名 具体的手机图标按钮名称或者实物名称
- 布局 图标之间的关系或者是页面内的位置信息
- 具体信息 图标具体有哪些信息，表达了什么信息
- 标签分析 为什么这个图标属于图片的分类标签

对于输入的全部样本提取总结信息：
- 出现图标: 不能超过10个,不出现具体样本独有信息，具体多少钱等用描述总结代替
- 分析原因: 为什么选取这些图标

要求输出格式:
<|>出现图标;出现图标;出现图标;出现图标;出现图标<|>分析原因<|>

#########################
例子1 Output:
<|>退货页面;退货商品信息;退货按钮;退款成功;退货运费未付提示;退货方式选择<|>这些图标通常出现在退货页面的核心部分，涉及退货流程的不同阶段，如商家同意退货、退货商品信息展示、选择退货方式、支付运费等。退货按钮是用户操作的核心入口，退款成功图标展示了最终状态，退货方式选择帮助用户选择合适的退货服务。这些图标属于典型的退货页面图标，因为它们提供了用户退货流程的指导、选择和确认功能。<|>
例子2 Output:
<|>支付费用详情;支付按钮;运费详情;选择支付方式;客服<|>这些图标是支付页面的核心组成部分，它们提供了用户完成支付所需的所有关键信息和操作选项。
例子3 Output:
<|>商品分类选项;商品图片;已选商品数量和状态<|>页面顶部区域作为用户首次接触网页内容的地方，通常用于放置导航栏和其他重要图标，以引导用户的浏览行为。<|>
######################
-Real Data-
######################
这些样本的标签是[{label}],他们在区域[{local}]出现了以下图标:{attribute}
######################
Output:
""" 