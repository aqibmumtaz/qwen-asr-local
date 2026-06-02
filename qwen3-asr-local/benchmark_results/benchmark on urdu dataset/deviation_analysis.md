# CommonVoice-Urdu 1000 — word deviation lists (E vs ground-truth B/C)

Every word below is a real token from **column E (ASR Roman output)**. Counts = occurrences in E.


# LIST 1 — words to ADD to lexicon (genuine ASR misspellings)

These E spellings are objectively wrong (ج→z, ق→q, ph→f, dropped/extra vowels). Target = correct Roman Urdu (project convention). **All already applied to data/lexicons.json.**


| count in E | ASR (E) | → correct |
|--:|---|---|
| 5 | gae | gaye |
| 5 | gai | gayi |
| 4 | khyaal | khayal |
| 4 | pahle | pehle |
| 3 | taraph | taraf |
| 3 | naaraaj | naraaz |
| 3 | jindagi | zindagi |
| 3 | mutaabik | mutaabiq |
| 3 | paakistani | Pakistani |
| 2 | shareeph | sharif |
| 2 | phalaan | falaan |
| 2 | phark | farq |
| 2 | gujar | guzar |
| 2 | saja | saza |
| 2 | kabja | qabza |
| 2 | jyada | zyada |
| 2 | kaayal | qaail |
| 2 | kaayam | qaaim |
| 2 | kaanoon | qaanoon |
| 2 | matalab | matlab |
| 2 | del | dil |
| 2 | beemari | bimari |
| 2 | madeena | madina |
| 2 | mumakin | mumkin |
| 2 | musalamaan | musalmaan |
| 2 | khatm | khatam |
| 1 | haaphij | hafiz |
| 1 | intajaar | intezar |
| 1 | zubaan | zaban |

# LIST 2 — words all-the-way INCORRECT vs ground truth B (NOT lexicon-fixable)


## 2A. Misheard words — 5 (model recognised a different word)


| count in E | ASR (E) |
|--:|---|
| 1 | media |
| 1 | tons |
| 1 | dete |
| 1 | the |
| 1 | two |

## 2B. Raw Arabic/Urdu script emitted — 140 types / 163 tokens (script-detection failure; post-processor never ran)


| count | token |
|--:|---|
| 5 | می |
| 5 | که |
| 4 | کی |
| 4 | کا |
| 3 | کو |
| 2 | مذهب |
| 2 | هستند |
| 2 | هم |
| 2 | پر |
| 2 | سے |
| 2 | کے |
| 2 | ختم |
| 1 | پاکستان |
| 1 | آسما |
| 1 | دوادر |
| 1 | مستقبل |
| 1 | کد |
| 1 | وی |
| 1 | سوابط |
| 1 | هوگر |
| 1 | باتر |
| 1 | اخبار |
| 1 | مجبورت |
| 1 | هریف |
| 1 | خلاف |
| 1 | شروع |
| 1 | کرده |
| 1 | حضرت |
| 1 | انس |
| 1 | رضی |
| 1 | اللہ |
| 1 | عنہ |
| 1 | از |
| 1 | آنهاهای |
| 1 | روایه |
| 1 | ته |
| 1 | خیالات |
| 1 | اللاباق |
| 1 | آپ |
| 1 | درجات |
| 1 | بلند |
| 1 | کرے |
| 1 | اقتصادام |
| 1 | شروعات |
| 1 | و |
| 1 | سماجی |
| 1 | مطالعه |
| 1 | دو |
| 1 | مختلف |
| 1 | علوم |
| 1 | بیشی |
| 1 | اشاره |
| 1 | زبانه |
| 1 | زده |
| 1 | آمی |
| 1 | زیادتر |
| 1 | غیرمارو |
| 1 | فنکار |
| 1 | khushagوار |
| 1 | این |
| 1 | تمام |
| 1 | اخراجات |
| 1 | خود |
| 1 | مزدور |
| 1 | زماده |
| 1 | khلاف |
| 1 | مصنوعی |
| 1 | گلها |
| 1 | سعی |
| 1 | خشبو |
| 1 | نهی |
| 1 | آتی |
| 1 | وفاقی |
| 1 | حکومت |
| 1 | پسکو |
| 1 | بلانگ |
| 1 | زمہ |
| 1 | داری |
| 1 | بر |
| 1 | ہونیادارے |
| 1 | دینے |
| 1 | فیصلہ |
| 1 | دانيش |
| 1 | تيمور |
| 1 | وحديات |
| 1 | كارا |
| 1 | البدر |
| 1 | ساد |
| 1 | غرامي |
| 1 | سيد |
| 1 | قاسم |
| 1 | محمود |
| 1 | آسان |
| 1 | بادل |
| 1 | چایه |
| 1 | های |
| 1 | خودشات |
| 1 | لاهک |
| 1 | هن |
| 1 | میرا |
| 1 | ارباب |
| 1 | اختیار |
| 1 | درخواست |
| 1 | کرنا |
| 1 | قدرت |
| 1 | یہ |
| 1 | خسین |
| 1 | نمونہ |
| 1 | ہونے |
| 1 | بچانا |
| 1 | لہور |
| 1 | سب |
| 1 | کچھ |
| 1 | لٹانے |
| 1 | باوجود |
| 1 | نون |
| 1 | لی |
| 1 | کیا |
| 1 | ملا |
| 1 | ضرورت |
| 1 | ا |
| 1 | س |
| 1 | عمر |
| 1 | ہے |
| 1 | زمانے |
| 1 | تغییر |
| 1 | دکھیے |
| 1 | shahريyaarahaan |
| 1 | بولينيشيا |
| 1 | چل |
| 1 | بازی |
| 1 | زین |
| 1 | آیا |
| 1 | خرید |
| 1 | لیا |
| 1 | بازیرازم |
| 1 | غیرضروری |
| 1 | تاکس |
| 1 | کردن |
| 1 | همایا |

# APPENDIX — benchmark-convention differences (NOT errors; do NOT add) — 82 words

Here the ASR (E) is already correct Roman Urdu; the benchmark (C) just uses a different style (writes w→o, e→i, inserts vowels e.g. `saamne`→`saamane`). Adding these would DEGRADE output.


| count in E | ASR (E) — correct | benchmark (C) — divergent |
|--:|---|---|
| 18 | kuch | kachh |
| 16 | ek | aik |
| 9 | kiya | kya |
| 8 | bahut | bahat |
| 7 | saamne | saamane |
| 5 | mauka | moqa |
| 5 | hamare | humare |
| 5 | kyon | kyun |
| 5 | inhein | anahin |
| 5 | karte | karate |
| 4 | sirf | saraf |
| 4 | mulk | malak |
| 4 | doosre | dosare |
| 4 | apne | apane |
| 4 | aalmi | aalami |
| 4 | maamla | maaamla |
| 3 | hoga | ga |
| 3 | shaamil | shaamal |
| 3 | doosra | dosara |
| 3 | yahan | yhaan |
| 3 | wahi | ohi |
| 3 | waapas | oaapas |
| 3 | baaten | baatin |
| 3 | kisi | kasi |
| 3 | hamein | hamin |
| 3 | duniya | daniaa |
| 3 | pasand | pasnad |
| 3 | dikhaai | dikhayi |
| 3 | yaad | iaad |
| 3 | doosri | dosari |
| 3 | nae | naie |
| 3 | yahi | ihi |
| 3 | wahan | whaan |
| 3 | sakti | sakati |
| 2 | mere | mire |
| 2 | chand | chaanad |
| 2 | deti | diti |
| 2 | baithe | bithe |
| 2 | chuka | chaka |
| 2 | bayaan | biaan |
| 2 | jamaaten | jmaatin |
| 2 | mausam | mosam |
| 2 | paidaawaar | pidaaoaar |
| 2 | khilaariyon | khlaarion |
| 2 | musabbat | masbat |
| 2 | sawaal | soaal |
| 2 | jahaan | jhaan |
| 2 | kar | ka |
| 2 | kahti | kahati |
| 2 | pahunch | pahnach |
| 2 | insani | anasani |
| 2 | sawaalaat | soalaat |
| 2 | jawaab | joaab |
| 2 | lete | lite |
| 2 | mujhse | majh |
| 2 | maahaul | maahol |
| 2 | khoobasoorti | khobasorati |
| 2 | pesh | pish |
| 2 | baawajood | baaojod |
| 2 | doosaron | dosaron |
| 2 | pas | paash |
| 2 | balki | balka |
| 2 | jamhooriyat | jamahorit |
| 2 | samajhne | samjhane |
| 2 | wale | oaale |
| 2 | kai | kaii |
| 2 | museebaton | masibaton |
| 2 | islaam | islam |
| 2 | sabse | sab |
| 2 | istemaal | astmaal |
| 2 | liye | laie |
| 2 | chalta | chalata |
| 2 | rahta | rahata |
| 2 | bete | bite |
| 2 | jaise | jise |
| 2 | karne | karane |
| 2 | chuke | chake |
| 2 | jagah | jaga |
| 2 | haalaat | halaat |
| 2 | rah | ra |
| 2 | hissa | hasa |
| 2 | amli | mali |