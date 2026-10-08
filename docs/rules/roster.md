# 角色名單

機器可讀的版本在 `docs/site/roster.json`（欄位 `six_standard`、`five` 等），網站的勾選清單由它產生。
來源：endfield.wiki.gg 的 Operator/List 頁面，2026-10-07 擷取，頁面標示資料截至 2026-09-01。歪池六星的中文名由專案作者提供。
艾爾黛拉、駿衛、萊萬汀（Laevatain）、潔爾佩塔（Gilberta）的譯名取自遊戲內公告；別禮、餘燼、黎風、提佛洛斯、梨諾是作者口述，可能與官方用字不同。其餘中文譯名尚未補上，稀有度也還沒逐一核對，發現錯誤請直接改 `roster.json`。

## 歪池的六星（1.6 上半限定池，共 7）

出六星而且歪掉時，從這 7 隻裡均等抽一隻。

| 英文 | 中文 | 身分 |
|---|---|---|
| Ardelia | 艾爾黛拉 | 常駐 |
| Last Rite | 別禮 | 常駐 |
| Ember | 餘燼 | 常駐 |
| Pogranichnik | 駿衛 | 常駐 |
| Lifeng | 黎風 | 常駐 |
| Typhoeus | 提佛洛斯 | 前期限定 |
| Liino | 梨諾 | 前期限定 |

常駐的 5 隻固定，後 2 隻是前兩期的限定角，每個限定池都會換。因為重複六星給的配額不看潛能，網站只對常駐 5 隻逐隻勾選，後 2 隻只問「你有幾隻」，換池時不用更新名單。復刻池的歪池名單尚未確認。

## 五星（共 10）

Alesh、Arclight、Avywenna、Chen Qianyu、Da Pan、Perlica、Purrchena、Snowshine、Wulfgard、Xaihi

## 其他六星（共 12）

Arcane、Argent Flow、Camille、Endministrator、Gilberta、Laevatain、Mi Fu、Rossi、Si、Tangtang、Yvonne、Zhuang Fangyi

## 四星（共 5）

Akekuri、Antal、Catcher、Estella、Fluorite

## 圖示

角色圖示是遊戲美術，著作權屬於發行商，所以不放進這個 repo。
`uv run python docs/site/fetch_icons.py` 會把圖示下載到 `docs/site/icons/`（已列入 .gitignore），之後建置網站時會自動嵌入勾選清單。
圖示網址的規則是 `https://endfield.wiki.gg/images/thumb/<名稱>_icon.png/90px-<名稱>_icon.png`，名稱中的空白換成底線。

## 用途

抽到已擁有的五星給 10 保障配額，已擁有的六星給 50，25 換 1 張通用憑證。
`gacharisk.analysis.rebate` 只需要「五星池裡已擁有的比例」與「歪池六星裡已擁有的比例」。
