# Infinite Retrieval

## Introduce
Implementation of currently Submitting Paper: "**Infinite Retrieval: Attention Enhanced LLMs in Long Context Processing**", which can apply any Transformer-based LLMs(Large Language Moidels) to handle unlimited length texts without training.

Notably，our method **InfiniRretri** can enbale the 0.5B(Qwen2.5-0.5B-Instruct), which originally had a maximum context length of 32K, to Haystack(retrieval) up over 1M tokens on Needle-In-a-Haystack(NIH) test, and even theoretically **infinite-length**.

![Origin](imgs/Qwen0.5B-FullKV-128k_0.446.png)
![Using InfiniRetri](imgs/Qwen0.5B-InfiniRetri_1m_1.000.png)


### Our Method Worflow

![](imgs/workflow.png)



## How to use
Firstly, you can only using pip install package ```infini-retri``` to get our method.
```python
pip install infini-retri==0.0.2
```

### Our Method Initialization
It's very convenient. You just need to pass in the model and its tokenizer directly, or you can simply passing in the model name or path. Additionally, it should be noted that our method can only using in tranditional **attention-based** Transformer, and the parameter of "attn_implementation" currently only using **"eager"**.

```python  
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
model_name_or_path = "Qwen/Qwen2.5-0.5B-Instruct" #  "./models/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, attn_implementation="eager") # attn_implementation only using "eager"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

from infini_retri import InfiniRetri
ir = InfiniRetri(model, tokenizer)
# ir = InfiniRetri(name_or_path=model_name_or_path) 
```

### Our Method in Model Inference
Our methodis to design an innovative processing mechanisum during the ***model inference(fix model.generate())*** to handle texts that exceed the upper limit of the original context length. Specifically, when use, there are there parameters of inputting text: `context`, `question`, `prompt`, as follows:
- `context:` (str) required option in the input text, just the complete contextual content that needs to be procssed(no include question and prompt).
- `question`: (str) option parameter,  passed in is your question for the LLMs, for example, the Question text in QA Pair is passed in this parameter. For all tasks, this parameter is recommended to be filled in, as it has a significant impact on the understanding of the LLMs and the ability to provide correct answers.
- `prompt`:  (str) option parameter, the instruction template that concatenates the `context` and `question` text sections above, especially noting that **when concatenating the `context` and `question` text sections, two "\n\n" are used to distinguish the boundaries of the text in each section. This is a necessary condition for our method to run normally**.  For example, its default prompt template is `"Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion:\n\n{question}\n\nAnswer:"`.
"

In addition, three parameters are provided here for everyone to adjust according to different types of tasks to achieve the best answer effect, as follows:

- `window_length`: (int, default 1024) this controls the length of the context window during the execution of our method. When setting , it only need to ensure that it is less than the maximum context window of the your using model.
- `topk`: (int, default 300) It affects the cache capacity size during the operation of our method and the actual length of context to be processed throughout the inference process. In theory, the larger the value, the larger the retrieval range during the operation. The actual optimal value depends on the user's problem handling and can be self adjusted.
- `answer_length`:(int, default 8) It affects the effectiveness of outputting the correct answer, and its value can be set based on the user's expected token length of the correct answer in the `context` section. In theory, the closer the token length is set to the correct answer in the `context`, the better the effect of the model's answer under our method.


```python

# This short passage is extracted from HarryPotter just present usage of our menthod. 
# Due to its short length, it cannot demonstrate the advantages of our method in handling task on ultra long text. 
context = """
Harry woke at five o'clock the next 
morning and was too excited and nervous to go back to sleep. He got up and pulled on his jeans because he didn't want to walk into the station in his wizard's robes — he'd change on the train. He checked his Hogwarts list yet again to make sure he had everything he needed, saw that Hedwig was shut safely in her cage, and then paced the room, waiting for the Dursleys to get up. Two hours later, Harry's huge, heavy trunk had been loaded into the Dursleys’ car, Aunt Petunia had talked Dudley into sitting next to Harry, and they had set off.They reached King's Cross at half past ten. Uncle Vernon dumped Harry's trunk onto a cart and wheeled it into the station for him. Harry thought this was strangely kind until Uncle Vernon stopped dead, facing the platforms with a nasty grin on his face.
"""  

question = "Why did Harry decide to wear jeans instead of his wizard's robes to the train station?"

prompt = "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion:\n\n{question}\n\nAnswer:" # Note "\n\n" in boundary.

response = ir.generate(context=context, question=question, prompt=prompt)
print("Response:", response)
```

## Advantage of Our Method

### Retrieval Task in Long Text
Our method has solved the Needle-In-a-Haystack (NIH) task for the first time compare prior cache kv methods, achieving the ability to fully retrieve the correct information in different LLMs in the NIH task. We strictly used the method from [KVcache Facotry](https://github.com/Zefan-Cai/KVCache-Factory/blob/main/run_needle_in_haystack.py) to test the NIH test on the PaulGrahamEssays dataset.

Comparison using Llama3-8B-Instruct in different KV Cache Compression methods in NIH Test:

![FullKV](imgs/Llama3-full-1k-32k_0.271.png)
![StreamingLLM](imgs/Llama3-streamingllm-1k-32k_0.121.png)
![H2O](imgs/Llama3-h2o-1k-32k_0.105.png)
![SnapKV](imgs/Llama3-snapkv-1k-32k_0.259.png)
![PyramidKV](imgs/Llama3-pyramidkv-1k-32k_0.283.png)
![InfiniRetri(Ours)](imgs/Llama3-infiniRetri-32k_0.951.png)



### Question-Answer Task in Long Text(Document, Book)

To read a long long book, and answer Question by user.

``` python

# This is the entire content from Chapter 29 of "Romance of the Three Kingdoms"
chapter = """
却说孙策自霸江东，兵精粮足。建安四年，袭取庐江，败刘勋，使虞翻驰檄豫章，豫章太守华歆投降。自此声势大振，乃遣张纮往许昌上表献捷。曹操知孙策强盛，叹曰：“狮儿难与争锋也！”遂以曹仁之女许配孙策幼弟孙匡，两家结婚。留张纮在许昌。孙策求为大司马，曹操不许。策恨之，常有袭许都之心。于是吴郡太守许贡，乃暗遣使赴许都上书于曹操。其略曰：“孙策骁勇，与项籍相似。朝廷宜外示荣宠，召在京师；不可使居外镇，以为后患。”使者赍书渡江，被防江将士所获，解赴孙策处。策观书大怒，斩其使，遣人假意请许贡议事。贡至，策出书示之，叱曰：“汝欲送我于死地耶！”命武士绞杀之。贡家属皆逃散。有家客三人，欲为许贡报仇，恨无其便。一日，孙策引军会猎于丹徒之西山，赶起一大鹿，策纵马上山逐之。正赶之间，只见树林之内有三个人持枪带弓面立。策勒马问曰：“汝等何人？”答曰：“乃韩当军士也。在此射鹿。”策方举辔欲行，一人拈枪望策左腿便刺。策大惊，急取佩剑从马上砍去，剑刃忽坠，止存剑靶在手。一人早拈弓搭箭射来，正中孙策面颊。策就拔面上箭，取弓回射放箭之人，应弦面倒。那二人举枪向孙策乱搠，大叫曰：“我等是许贡家客，特来为主人报仇！”策别无器械，只以弓拒之，且拒且走。二人死战不退。策身被数枪，马亦带伤。正危急之时，程普引数人至。孙策大叫：“杀贼！“程普引众齐上，将许贡家客砍为肉泥。看孙策时，血流满面，被伤至重，乃以刀割抱，裹其伤处，救回吴会养病。后人有诗赞许家三客曰：“孙郎智勇冠江湄，射猎山中受困危。许客三人能死义，杀身豫让未为奇。”却说孙策受伤而回，使人寻请华伦医治。不想华佗已往中原去了，止有徒弟在吴，命其治疗。其徒曰：“箭头有药，毒已入骨。须静养百日，方可无虞。若怒气冲激，其疮难治。”孙策为人最是性急，恨不得即日便愈。将息到二十余日，忽闻张纮有使者自许昌回，策唤问之。使者曰：“曹操甚惧主公；其帐下谋士，亦俱敬服；惟有郭嘉不服。”策曰：“郭嘉曾有何说？”使者不敢言。策怒，固问之。使者只得从实告曰：“郭嘉曾对曹操言主公不足惧也：轻而无备，性急少谋，乃匹夫之勇耳，他日必死于小人之手。”策闻言，大怒曰：“匹夫安敢料吾！吾誓取许昌！”遂不待疮愈，便欲商议出兵。张昭谏曰：“医者戒主公百日休动，今何因一时之忿，自轻万金之躯？”正话间，忽报袁绍遣使陈震至。策唤入问之。震具言袁绍欲结东吴为外应，共攻曹操。策大喜，即日会诸将于城楼上，设宴款待陈震。饮酒之间，忽见诸将互相耳语，纷纷下楼。策怪问何故，左右曰：“有于神仙者，今从楼下过，诸将欲往拜之耳。”策起身凭栏观之，见一道人，身披鹤氅，手携藜杖，立于当道，百姓俱焚香伏道而拜。策怒曰：“是何妖人？快与我擒来！”左右告曰：“此人姓于，名吉，寓居东方，往来吴会，普施符水，救人万病，无有不验。当世呼为神仙，未可轻渎。”策愈怒，喝令：“速速擒来！违者斩！”

左右不得已，只得下楼，拥于吉至楼上。策叱曰：“狂道怎敢煽惑人心！”于吉曰：“贫道乃琅琊宫道士，顺帝时曾入山采药，得神书于阳曲泉水上，号曰《太平青领道》，凡百余卷，皆治人疾病方术。贫道得之，惟务代天宣化，普救万人，未曾取人毫厘之物，安得煽惑人心？”策曰：“汝毫不取人，衣服饮食，从何而得？汝即黄巾张角之流，今若不诛，必为后患！”叱左右斩之。张昭谏曰：“于道人在江东数十年，并无过犯，不可杀害。”策曰：“此等妖人，君杀之，何异屠猪狗！”众官皆苦谏，陈震亦劝。策怒未息，命且囚于狱中。众官俱散。陈震自归馆驿安歇。孙策归府，早有内侍传说此事与策母吴太夫人知道。夫人唤孙策入后堂，谓曰：“吾闻汝将于神仙下于缧绁。此人多曾医人疾病，军民敬仰，不可加害。”策曰：“此乃妖人，能以妖术惑众，不可不除！”夫人再三劝解。策曰：“母亲勿听外人妄言，儿自有区处。乃出唤狱吏取于吉来问。原来狱吏皆敬信于吉，吉在狱中时，尽去其枷锁；及策唤取，方带枷锁而出。策访知大怒，痛责狱吏，仍将于吉械系下狱。张昭等数十人，连名作状，拜求孙策，乞保于神仙。策曰：“公等皆读书人，何不达理？昔交州刺史张津，听信邪教，鼓瑟焚香，常以红帕裹头，自称可助出军之威，后竟为敌军所杀。此等事甚无益，诸君自未悟耳。吾欲杀于吉，正思禁邪觉迷也。”

吕范曰：“某素知于道人能祈风祷雨。方今天旱，何不令其祈雨以赎罪？”策曰：“吾且看此妖人若何。”遂命于狱中取出于吉，开其枷锁，令登坛求雨。吉领命，即沐浴更衣，取绳自缚于烈日之中。百姓观者，填街塞巷。于吉谓众人曰：“吾求三尺甘霖，以救万民，然我终不免一死。”众人曰：“若有灵验，主公必然敬服。”于吉曰：“气数至此，恐不能逃。”少顷，孙策亲至坛中下令：“若午时无雨，即焚死于吉。”先令人堆积干柴伺候。将及午时，狂风骤起。风过处，四下阴云渐合。策曰：“时已近午，空有阴云，而无甘雨，正是妖人！”叱左右将于吉扛上柴堆，四下举火，焰随风起。忽见黑烟一道，冲上空中，一声响喨，雷电齐发，大雨如注。顷刻之间，街市成河，溪涧皆满，足有三尺甘雨。于吉仰卧于柴堆之上，大喝一声，云收雨住，复见太阳。于是众官及百姓，共将于吉扶下柴堆，解去绳索，再拜称谢。孙策见官民俱罗拜于水中，不顾衣服，乃勃然大怒，叱曰：“晴雨乃天地之定数，妖人偶乘其便，你等何得如此惑乱！”掣宝剑令左右速斩于吉。众官力谏，策怒曰：“尔等皆欲从于吉造反耶！”众官乃不敢复言。策叱武士将于吉一刀斩头落地。只见一道青气，投东北去了。策命将其尸号令于市，以正妖妄之罪。

是夜风雨交作，及晓，不见了于吉尸首。守尸军士报知孙策。策怒，欲杀守尸军士。忽见一人，从堂前徐步而来，视之，却是于吉。策大怒，正欲拔剑斫之，忽然昏倒于地。左右急救入卧内，半晌方苏。吴太夫人来视疾，谓策曰：“吾儿屈杀神仙，故招此祸。”策笑曰：“儿自幼随父出征，杀人如麻，何曾有为祸之理？今杀妖人，正绝大祸，安得反为我祸？”夫人曰：“因汝不信，以致如此；今可作好事以禳之。”策曰：“吾命在天，妖人决不能为祸。何必禳耶！”夫人料劝不信，乃自令左右暗修善事禳解。是夜二更，策卧于内宅，忽然阴风骤起，灯灭而复明。灯影之下，见于吉立于床前。策大喝曰：“吾平生誓诛妖妄，以靖天下！汝既为阴鬼，何敢近我！”取床头剑掷之，忽然不见。吴太夫人闻之，转生忧闷。策乃扶病强行，以宽母心。母谓策曰：“圣人云：‘鬼神之为德，其盛矣乎！’又云：‘祷尔于上下神袛。’鬼神之事，不可不信。汝屈杀于先生，岂无报应？吾已令人设醮于郡之玉清观内，汝可亲往拜祷，自然安妥。”

策不敢违母命，只得勉强乘轿至玉清观。道士接入，请策焚香，策焚香而不谢。忽香炉中烟起不散，结成一座华盖，上面端坐着于吉。策怒，唾骂之；走离殿宇，又见于吉立于殿门首，怒目视策。策顾左右曰：“汝等见妖鬼否？”左右皆云未见。策愈怒，拔佩剑望于吉掷去，一人中剑而倒。众视之，乃前日动手杀于吉之小卒，被剑斫入脑袋，七窍流血而死。策命扛出葬之。比及出观，又见于吉走入观门来。策曰：“此观亦藏妖之所也！”遂坐于观前，命武士五百人拆毁之。武士方上屋揭瓦，却见于吉立于屋上，飞瓦掷地。策大怒，传令逐出本观道士，放火烧毁殿宇。火起处，又见于吉立于火光之中。策怒归府，又见于吉立于府门前。策乃不入府，随点起三军，出城外下寨，传唤众将商议，欲起兵助袁绍夹攻曹操。众将俱曰：“主公玉体违和，未可轻动。且待平愈，出兵未迟。”是夜孙策宿于寨内，又见于吉披发而来。策于帐中叱喝不绝。次日，吴太夫人传命，召策回府。策乃归见其母。夫人见策形容憔悴，泣曰：“儿失形矣！”策即引镜自照，果见形容十分瘦损，不觉失惊，顾左右曰：“吾奈何憔悴至此耶！”言未已，忽见于吉立于镜中。策拍镜大叫一声，金疮迸裂，昏绝于地。夫人令扶入卧内。须臾苏醒，自叹曰：“吾不能复生矣！”

随召张昭等诸人，及弟孙权，至卧榻前，嘱付曰：“天下方乱，以吴越之众，三江之固，大可有为。子布等幸善相吾弟。”乃取印绶与孙权曰：“若举江东之众，决机于两阵之间，与天下争衡，卿不如我；举贤任能，使各尽力以保江东，我不如卿。卿宜念父兄创业之艰难，善自图之！”权大哭，拜受印绶。策告母曰：“儿天年已尽，不能奉慈母。今将印绶付弟，望母朝夕训之。父兄旧人，慎勿轻怠。”母哭曰：“恐汝弟年幼，不能任大事，当复如何？”策曰：“弟才胜儿十倍，足当大任。倘内事不决，可问张昭；外事不决，可问周瑜。恨周瑜不在此，不得面嘱之也！”又唤诸弟嘱曰：“吾死之后，汝等并辅仲谋。宗族中敢有生异心者，众共诛之；骨肉为逆，不得入祖坟安葬。”诸弟泣受命。又唤妻乔夫人谓曰：“吾与汝不幸中途相分，汝须孝养尊姑。早晚汝妹入见，可嘱其转致周郎，尽心辅佐吾弟，休负我平日相知之雅。”言讫，瞑目而逝。年止二十六岁。后人有诗赞曰：“独战东南地，人称小霸王。运筹如虎踞，决策似鹰扬。威镇三江靖，名闻四海香。临终遗大事，专意属周郎。”

孙策既死，孙权哭倒于床前。张昭曰：“此非将军哭时也。宜一面治丧事，一面理军国大事。”权乃收泪。张昭令孙静理会丧事，请孙权出堂，受众文武谒贺。孙权生得方颐大口，碧眼紫髯。昔汉使刘琬入吴，见孙家诸昆仲，因语人曰：“吾遍观孙氏兄弟，虽各才气秀达，然皆禄祚不终。惟仲谋形貌奇伟，骨格非常，乃大贵之表，又亨高寿，众皆不及也。”

且说当时孙权承孙策遗命，掌江东之事。经理未定，人报周瑜自巴丘提兵回吴。权曰：“公瑾已回，吾无忧矣。”原来周瑜守御巴丘。闻知孙策中箭被伤，因此回来问候；将至吴郡，闻策已亡，故星夜来奔丧。当下周瑜哭拜于孙策灵柩之前。吴太夫人出，以遗嘱之语告瑜，瑜拜伏于地曰：“敢不效犬马之力，继之以死！”少顷，孙权入。周瑜拜见毕，权曰：“愿公无忘先兄遗命。”瑜顿首曰：“愿以肝脑涂地，报知己之恩。”权曰：“今承父兄之业，将何策以守之？”瑜曰：“自古得人者昌，失人者亡。为今之计，须求高明远见之人为辅，然后江东可定也。”权曰：“先兄遗言：内事托子布，外事全赖公瑾。”瑜曰：“子布贤达之士，足当大任。瑜不才，恐负倚托之重，愿荐一人以辅将军。”权问何人。瑜曰：“姓鲁，名肃，字子敬，临淮东川人也。此人胸怀韬略，腹隐机谋。早年丧父，事母至孝。其家极富，尝散财以济贫乏。瑜为居巢长之时，将数百人过临淮，因乏粮，闻鲁肃家有两囷米，各三千斛，因往求助。肃即指一囷相赠，其慷慨如此。平生好击剑骑射，寓居曲阿。祖母亡，还葬东城。其友刘子扬欲约彼往巢湖投郑宝，肃尚踌躇未往。今主公可速召之。”权大喜，即命周瑜往聘。

瑜奉命亲往，见肃叙礼毕，具道孙权相慕之意。肃曰：“近刘子扬约某往巢湖，某将就之。”瑜曰：“昔马援对光武云：当今之世，非但君择臣，臣亦择君。今吾孙将军亲贤礼士，纳奇录异，世所罕有。足下不须他计，只同我往投东吴为是。”

肃从其言，遂同周瑜来见孙权。权甚敬之，与之谈论，终日不倦。一日，众官皆散，权留鲁肃共饮，至晚同榻抵足而卧。夜半，权问肃曰：“方今汉室倾危，四方纷扰；孤承父兄余业，思为桓、文之事，君将何以教我？”肃曰：“昔汉高祖欲尊事义帝而不获者，以项羽为害也。今之曹操可比项羽，将军何由得为桓、文乎？肃窃料汉室不可复兴，曹操不可卒除。为将军计，惟有鼎足江东以观天下之衅。今乘北方多务，剿除黄祖，进伐刘表，竟长江所极而据守之；然后建号帝王，以图天下：此高祖之业也。”权闻言大喜，披衣起谢。次日厚赠鲁肃，并将衣服帏帐等物赐肃之母。

肃又荐一人见孙权：此人博学多才，事母至孝；覆姓诸葛，名瑾，字子瑜，琅琊南阳人也。权拜之为上宾。瑾劝权勿通袁绍，且顺曹操，然后乘便图之。权依言，乃遣陈震回，以书绝袁绍。却说曹操闻孙策已死，欲起兵下江南。侍御史张纮谏曰：“乘人之丧而伐之，既非义举；若其不克，弃好成仇：不如因而善遇之。”操然其说，乃即奏封孙权为将军，兼领会稽太守；即令张纮为会稽都尉，赍印往江东。孙权大喜，又得张纮回吴，即命与张昭同理政事。张纮又荐一人于孙权：此人姓顾，名雍，字元叹，乃中郎蔡邕之徒；其为人少言语，不饮酒，严厉正大。权以为丞，行太守事。自是孙权威震江东，深得民心。且说陈震回见袁绍，具说：“孙策已亡，孙权继立。曹操封之为将军，结为外应矣。”袁绍大怒，遂起冀、青、幽、并等处人马七十余万，复来攻取许昌。正是：江南兵革方休息，冀北干戈又复兴。未知胜负若何，且听下文分解。 
"""


from infini_retri import InfiniRetri

model_name_or_path = "Qwen/Qwen2.5-0.5B-Instruct" #  "./models/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, attn_implementation="eager") # attn_implementation only using "eager"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ir = InfiniRetri(model, tokenizer)


query = "孙策离世前向孙权嘱咐，内事不决可以问谁？外事不决可以问谁？"
template = "阅读以下书籍然后回答问题。\n\n{context}\n\n问题：\n\n{question}\n\n答案：" # Note "\n\n" in boundary.

response = ir.generate(context=chapter, question=query, prompt=template)
print("Response:", response)
#  Qwen2.5-0.5B using our method: 孙策离世前向孙权嘱咐，内事不决可以问张昭；外事不决可以问周瑜。
```
