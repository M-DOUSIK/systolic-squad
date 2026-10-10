"""Unlabeled indoor images with people / close objects for teacher distillation (Wikimedia Commons, free licences only).
python fetch_unlabeled.py -> img_###.jpg (640 px) + LICENSES.md"""
import json, os, re, urllib.parse, urllib.request
UA = {"User-Agent": "SystolicSquad-hackathon/1.0 (education; depth-estimation training images)"}
OUT = os.path.dirname(os.path.abspath(__file__))
QUERIES = ["selfie indoor", "person using laptop", "video call webcam", "people working office desk", "students classroom",
           "family living room", "person sitting desk computer", "man portrait indoor", "woman portrait indoor", "hand holding phone",
           "person standing hallway", "people meeting room", "child playing room", "person kitchen cooking", "group photo indoor",
           "hackathon participants", "laboratory students", "library reading people", "cafe people table", "person close up face"]
import sys, glob
if len(sys.argv) > 1 and sys.argv[1] == "2":                       # second batch: more close-range people / hands / desks
    QUERIES = ["person holding object close camera", "student at desk laptop", "people talking indoor", "man sitting chair room",
               "woman sitting office", "teenagers classroom", "workshop participants table", "engineer laboratory bench",
               "people at conference table", "person reading book indoor", "girl studying desk", "boy using computer",
               "office cubicle worker", "people standing corridor", "hands keyboard", "person pointing indoor",
               "customer shop counter", "nurse hospital room", "teacher whiteboard classroom", "coworking space people"]
if len(sys.argv) > 1 and sys.argv[1] == "3":                       # third batch: environments and objects (furniture, rooms, close objects)
    QUERIES = ["office chair", "wooden chair room", "dining table chairs", "desk with computer", "classroom desks chairs", "conference room table",
               "living room sofa", "bookshelf room", "kitchen counter interior", "corridor interior", "staircase interior", "office interior",
               "laboratory interior", "computer lab room", "library interior shelves", "bedroom interior", "cafe interior tables",
               "workshop interior", "storage shelves boxes", "auditorium seats", "hospital room interior", "lobby interior",
               "chair close up", "laptop on table", "cardboard box floor", "backpack on floor", "potted plant indoor", "whiteboard room",
               "trash bin indoor", "door hallway"]
idx = [int(re.findall(r"img_(\d+)", f)[0]) for f in glob.glob(os.path.join(OUT, "img_*.jpg")) + glob.glob(os.path.join(OUT, "rejected", "img_*.jpg"))]
rows, n, seen = [], max(idx + [999]) + 1, set()
for q in QUERIES:
    p = {"action": "query", "generator": "search", "gsrsearch": f"{q} filetype:bitmap", "gsrnamespace": "6", "gsrlimit": "40",
         "prop": "imageinfo", "iiprop": "url|extmetadata|mime|size", "iiurlwidth": "640", "format": "json"}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(p), headers=UA), timeout=40))
    except Exception as e:
        print(q, e); continue
    took = 0
    for pg in d.get("query", {}).get("pages", {}).values():
        ii = pg["imageinfo"][0]
        if ii.get("mime") != "image/jpeg" or "thumburl" not in ii or pg["title"] in seen: continue
        if ii.get("width", 0) < 640 or ii.get("height", 0) < 400: continue
        md = ii.get("extmetadata", {})
        lic = md.get("LicenseShortName", {}).get("value", "")
        if not re.search(r"CC|Public domain|PD", lic, re.I) or re.search(r"NC|ND", lic): continue
        fn = f"img_{n:03d}.jpg"
        try:
            open(os.path.join(OUT, fn), "wb").write(urllib.request.urlopen(urllib.request.Request(ii["thumburl"], headers=UA), timeout=40).read())
        except Exception as e:
            continue
        seen.add(pg["title"])
        artist = re.sub("<[^>]+>", "", md.get("Artist", {}).get("value", "unknown")).strip().replace("|", "/")[:80]
        rows.append(f"| {fn} | {pg['title'].replace('|', '/')} | {artist} | {lic} | {ii['descriptionurl']} |")
        n += 1; took += 1
        if took >= 18: break
    print(q, took, flush=True)
lic_path = os.path.join(OUT, "LICENSES.md")
if os.path.exists(lic_path):                                       # second batch: append, keep the first batch's rows
    open(lic_path, "a", encoding="utf-8").write("\n".join(rows) + "\n")
else:
    open(lic_path, "w", encoding="utf-8").write(
        "# Unlabeled training images (Wikimedia Commons, free licences, no NC/ND) - inputs for teacher distillation only\n\n"
        "| File | Commons title | Author | Licence | Source |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
print(n, "images")
