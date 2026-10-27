# Darwish LPR — قارئ لوحات السيارات المصرية

نظام بيتوصل بكاميرات المراقبة (Hikvision NVR أو أي كاميرا RTSP) وبيطلّع أرقام لوحات العربيات اللي معدية في الشارع. كل لوحة بتتسجل بالوقت والكاميرا وصورة، وفي لوحة تحكم ويب بالعربي فيها بحث وتنبيهات.

```
NVR / كاميرات ──RTSP──► قراية فريمات (thread لكل كاميرا، إعادة اتصال تلقائي)
    ► YOLOv8 + ByteTrack: كشف العربيات ومتابعة كل عربية
    ► موديل اللوحات: يلاقي اللوحة ► موديل الحروف: يقرا كل حرف ورقم ► قواعد اللوحة المصرية
    ► تصويت على القراية عبر كذا فريم لكل عربية ► حدث واحد لكل عربية
    ► SQLite + صور ► لوحة تحكم ويب + تنبيهات (webhook)
```

## المميزات

- **كاميرات Hikvision**: تكتب رقم القناة بس، والنظام يبني رابط RTSP. وفي أداة بتكتشف القنوات الشغالة.
- **أي مصدر تاني**: رابط RTSP أو HTTP، كاميرا USB، أو ملف فيديو.
- **لوحات مصرية**: موديلات متدربة على لوحات مصرية حقيقية (داتاسيت EALPR). بتقرا الحروف والأرقام العربي بترتيبها الصح: الحروف من اليمين والأرقام من الشمال.
- **دقة أعلى بالتصويت**: كل عربية بتتقري في كذا فريم، والقراية اللي بتتكرر أكتر هي اللي بتتسجل، ومن غير تكرار.
- **لوحة تحكم بالعربي**: بث مباشر، وآخر اللوحات لحظة بلحظة، وسجل فيه بحث وفلترة وتصدير CSV.
- **قائمة مراقبة**: تنبيه أحمر في اللوحة، وإرسال webhook (n8n أو واتساب أو غيره) لما لوحة معينة تعدي.
- **خصوصية**: باسورد على كل الصفحات، ومسح تلقائي للبيانات القديمة (30 يوم افتراضيًا).
- يشتغل على **GPU (NVIDIA)** أو **CPU** تلقائيًا.

## التشغيل

### 1. التسطيب (Python 3.10+)

```bash
git clone https://github.com/mohamedstar19/Darwish_LPR.git
cd Darwish_LPR
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux: source .venv/bin/activate
pip install -r requirements.txt
```

> لو عندك كارت NVIDIA: سطّب نسخة CUDA من torch الأول من [pytorch.org](https://pytorch.org/get-started/locally/)، وبعدين `requirements.txt`.

### 2. الإعدادات

```bash
cp config.example.yaml config.yaml      # Windows: copy config.example.yaml config.yaml
```

عدّل في `config.yaml` عنوان الـ NVR واليوزر والقنوات. الباسوردات بتتحط في متغيرات بيئة، مش في الملف:

```bash
# Linux
export NVR_PASSWORD='...'  LPR_WEB_PASSWORD='...'
# Windows (PowerShell)
$env:NVR_PASSWORD='...'; $env:LPR_WEB_PASSWORD='...'
```

### 3. اختبار الكاميرات

```bash
python -m lpr scan-nvr --channels 16   # أنهي قنوات شغالة
python -m lpr probe                    # صورة من كل كاميرا في data/probe/
```

### 4. التشغيل

```bash
python -m lpr run
```

افتح `http://IP-الجهاز:8000` وادخل بيوزر وباسورد الويب.

### تجربة على فيديو متسجل (من غير كاميرات)

```bash
python -m lpr video samples/street.mp4 --json results.json
```

## الأداء المتوقع

| الجهاز | الكاميرات (6 فريم/ثانية لكل كاميرا) |
|---|---|
| CPU حديث (i5/i7 جيل 10+) | 1–2 |
| GPU (مثلًا RTX 3060) | 6–10 |
| Jetson Orin Nano | 2–4 |

قلّل `process_fps` واستخدم `roi` لو الجهاز مش ملاحق.

## الدقة

تم القياس على عربيات من داتاسيت EALPR الموديلات ما شافتهاش في التدريب، بصورة واحدة لكل عربية ومن غير تصويت:

<!-- ACCURACY -->

في التشغيل الحقيقي، كل عربية بتتقري في كذا فريم وبيحصل تصويت، فالنتيجة بتبقى أحسن من الصورة الواحدة. **بس صور EALPR متصورة بالموبايل من قريب**، وكاميرات المراقبة مختلفة. عشان كده:

- **مكان الكاميرا هو أهم عامل.** عرض اللوحة لازم يكون 130 بكسل على الأقل. الكاميرا الواسعة اللي بتصور الشارع كله مش هتقرا لوحات. راجع [docs/hikvision.md](docs/hikvision.md).
- **أكبر تحسين بعد كده**: درّب الموديلات تاني بصور من كاميراتك. الصور بتتجمع لوحدها في `data/snapshots/`. راجع [docs/models.md](docs/models.md).

## هيكل المشروع

| الملف | الوظيفة |
|---|---|
| `lpr/camera.py` | قراية RTSP/USB/فيديو وإعادة الاتصال |
| `lpr/detector.py` | كشف العربيات ومتابعتها، وكشف اللوحة |
| `lpr/ocr.py` | قراية الحروف والأرقام وترتيبها في لوحة |
| `lpr/plate_text.py` | قواعد اللوحة المصرية (تطبيع وتحقق) |
| `lpr/voting.py` | التصويت ومنع التكرار |
| `lpr/storage.py` | قاعدة البيانات والصور والمسح التلقائي |
| `lpr/pipeline.py` | ربط كل حاجة لكل كاميرا، والتنبيهات |
| `lpr/web.py` + `lpr/static/` | لوحة التحكم والـ API |
| `models/` | موديلات اللوحات والحروف المدرّبة |
| `tools/` | تجهيز الداتا والتدريب وقياس الدقة |

### الـ API (كل الطلبات بـ HTTP Basic)

| الطلب | الوصف |
|---|---|
| `GET /api/events?q=&camera=&since=&until=&limit=` | السجل والبحث |
| `GET /api/events/stream` | اللوحات الجديدة لحظيًا (SSE) |
| `GET /api/events.csv` | تصدير |
| `GET /api/cameras` | حالة الكاميرات |
| `GET /api/cameras/{id}/live.mjpg` | البث المباشر مع المربعات |
| `GET/POST/DELETE /api/watchlist` | قائمة المراقبة |

شكل الـ webhook:

```json
{"type": "plate", "event": {"plate_display": "ب ط ص ١٢٣٤", "camera_name": "البوابة", "ts": 1767225600.0, "watch_kind": "alert", "...": "..."}}
```

## الاختبارات

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## الخطوات الجاية

1. إعادة تدريب الموديلات على صور كاميراتك (`tools/train.py`).
2. تشغيل كخدمة (Windows Service / systemd) وDocker.
3. ربط بأنظمة تانية: بوابات، حضور عربيات، أو تنبيه واتساب.

## الترخيص

كود المشروع MIT. الفكرة الأولى مأخوذة من [LPR-Vehicle-License-Detection](https://github.com/Nawaf-Rayhan585/LPR-Vehicle-License-Detection) (MIT)، والكود اتكتب من جديد.

الموديلات في `models/` متدربة على [EALPR](https://github.com/ahmedramadan96/EALPR) (Youssef et al., ACIRS 2022). الداتاسيت مفيهاش ملف ترخيص، فراجع [docs/models.md](docs/models.md) قبل أي استخدام تجاري.

> ⚠️ أرقام اللوحات بيانات شخصية. استخدم النظام في حدود القانون وسياسة الخصوصية بتاعتك.
