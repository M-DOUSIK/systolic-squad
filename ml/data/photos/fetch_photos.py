"""Fetch ~12 indoor photos from Wikimedia Commons (free licences), 640 px wide, and write LICENSES.md (title, author, licence, URL)."""
import json, os, re, urllib.parse, urllib.request
UA = {"User-Agent": "SystolicSquad-hackathon/1.0 (education; depth-estimation test images)"}
CATS = ["Category:Corridors", "Category:Living rooms", "Category:Offices", "Category:Kitchens", "Category:Classrooms", "Category:Stairs"]
OUT = os.path.dirname(os.path.abspath(__file__))
rows, n = [], 0
for cat in CATS:
    q = {"action": "query", "generator": "categorymembers", "gcmtitle": cat, "gcmtype": "file", "gcmlimit": "12", "prop": "imageinfo",
         "iiprop": "url|extmetadata|mime", "iiurlwidth": "640", "format": "json"}
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(q)
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))
    except Exception as e:
        print(cat, e); continue
    took = 0
    for p in d.get("query", {}).get("pages", {}).values():
        ii = p["imageinfo"][0]
        if ii.get("mime") != "image/jpeg" or "thumburl" not in ii: continue
        md = ii.get("extmetadata", {})
        lic = md.get("LicenseShortName", {}).get("value", "")
        if not re.search(r"CC|Public domain|PD", lic, re.I): continue
        fn = f"web_{n:02d}.jpg"
        try:
            open(os.path.join(OUT, fn), "wb").write(urllib.request.urlopen(urllib.request.Request(ii["thumburl"], headers=UA), timeout=30).read())
        except Exception as e:
            print(e); continue
        artist = re.sub("<[^>]+>", "", md.get("Artist", {}).get("value", "unknown")).strip()
        rows.append(f"| {fn} | {p['title']} | {artist} | {lic} | {ii['descriptionurl']} |")
        n += 1; took += 1
        if took == 2: break
open(os.path.join(OUT, "LICENSES.md"), "w", encoding="utf-8").write(
    "# Test photos (Wikimedia Commons, free licences) - used only as test inputs for the depth model (accuracy report, golden vectors)\n\n"
    "| File | Commons title | Author | Licence | Source |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
print(n, "photos")
