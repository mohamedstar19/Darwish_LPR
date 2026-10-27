# الموديلات

النظام بيستخدم 3 موديلات YOLO:

| الموديل | الملف | بيعمل إيه |
|---|---|---|
| العربيات | `yolov8n.pt` (بيتنزل لوحده) | يلاقي العربيات والموتوسيكلات والأتوبيسات والنقل، ويتابع كل عربية |
| اللوحات | `models/egypt_plate.pt` | يلاقي اللوحة جوه صورة العربية |
| الحروف | `models/egypt_chars.pt` | يتعرف على كل حرف ورقم في اللوحة (17 حرف + 10 أرقام) |

بعد كده الحروف بتترتب من اليمين للشمال والأرقام من الشمال لليمين، وبتتطبق قواعد شكل اللوحة المصرية. كل عربية بتتقري في كذا فريم وبيحصل تصويت على القراية.

> جربنا EasyOCR الأول وجاب **2.5% بس** قرايات صح على لوحات مصرية حقيقية، فاتشال.

## الداتا اللي اتدربت عليها

الموديلين اتدربوا على **EALPR**، ودي داتاسيت عامة فيها حوالي 2000 صورة عربية مصرية:

> Youssef, Sayed & Ali, *A New Benchmark Dataset for Egyptian License Plate Detection and Recognition*, ACIRS 2022. https://github.com/ahmedramadan96/EALPR

الداتاسيت مفيهاش ملف ترخيص، وأصحابها بيطلبوا الاستشهاد بيها. لو هتستخدم النظام تجاريًا، اتأكد من الترخيص معاهم أو درّب على صورك أنت.

## النتايج

شوف قسم «الدقة» في [README](../README.md). تقيس النتايج بنفسك كده:

```bash
git clone --depth 1 https://github.com/ahmedramadan96/EALPR datasets/EALPR
python tools/evaluate.py datasets/EALPR
```

## التدريب على صور كاميراتك (أهم خطوة لرفع الدقة)

صور EALPR أغلبها متصورة بالموبايل من قريب. كاميرات المراقبة زاويتها وإضاءتها مختلفة، والدقة بتعلى كتير لما تضيف صور منها:

1. شغّل النظام كام يوم، والصور هتتجمع في `data/snapshots/` (صورة للعربية وصورة للوحة).
2. علّم عليها بأداة زي [Label Studio](https://labelstud.io/) أو CVAT بصيغة YOLO:
   - صور العربيات: class واحد `plate`.
   - صور اللوحات: 27 class بنفس ترتيب `classes.txt` اللي في EALPR.
3. ضيفها لفولدرات `datasets/yolo/plates` و`datasets/yolo/chars`، وبعدين:

```bash
python tools/prepare_ealpr.py datasets/EALPR datasets/yolo   # مرة واحدة
python tools/train.py plates --data datasets/yolo/plates/data.yaml --device 0
python tools/train.py chars  --data datasets/yolo/chars/data.yaml  --device 0
```

الموديلات الجديدة بتتحفظ في `models/` وبتشتغل من غير ما تغيّر أي إعداد. على GPU التدريب بياخد دقايق، وعلى CPU ساعة أو اتنين.
