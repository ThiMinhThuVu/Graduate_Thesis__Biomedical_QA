import os
from utils.util import *

file_crawl_1 = r"data\cache\crawl_wikipedia\0a0bc4f22da7f01e0515da6981f11545.json"
file_crawl_2 = r"data\cache\crawl_wikipedia_tool_2\0a0bc4f22da7f01e0515da6981f11545.json"

link = "https://en.wikipedia.org/wiki/epigenetic_clock"

if os.path.exists(file_crawl_1):
    crawl_wiki = load_json(file_crawl_1)
    if crawl_wiki and link:
        crawled_data = crawl_wiki.get('crawl_wiki', {}).get('html_page').get(link, None)
        if crawled_data and 'content' in crawled_data and len(crawled_data['content']) != 0:
            wiki_content = wiki_pre_processing(crawled_data)
            save_json(wiki_content, 'tool_1.json')
            
if os.path.exists(file_crawl_2):
    crawl_wiki = load_json(file_crawl_2)
    if crawl_wiki and link:
        crawled_data = crawl_wiki.get('crawl_wiki', {}).get('html_page').get(link, None)
        if crawled_data and 'content' in crawled_data and len(crawled_data['content']) != 0:
            wiki_content = wiki_pre_processing_tool_2(crawled_data)
            save_json(wiki_content, 'tool_2.json')

# block = "The strong correlation between aging and DNA methylation levels has been known since the late 1960s. A vast literature describes sets of CpGs whose DNA methylation levels correlate with age. The first robust demonstration that DNA methylation levels in saliva could generate age predictors with an average accuracy of 5.2 years was published by a UCLA team including Sven Bocklandt, Steve Horvath, and Eric Vilain in 2011 (Bocklandt et al. 2011). The laboratories of Trey Ideker and Kang Zhang at the University of California, San Diego published the Hannum epigenetic clock (Hannum 2013), which consisted of 71 markers that accurately estimate age based on blood methylation levels. The first multi-tissue epigenetic clock, Horvath's epigenetic clock, was developed by Steve Horvath, a professor of human genetics  and biostatistics at UCLA (Horvath 2013). Horvath spent over 4 years collecting publicly available Illumina DNA methylation data and identifying suitable statistical methods.\nThe age estimator was developed using 8,000 samples from 82 Illumina DNA methylation array datasets, encompassing 51 healthy tissues and cell types. The major innovation of Horvath's epigenetic clock lies in its wide applicability: the same set of 353 CpGs and the same prediction algorithm is used irrespective of the DNA source within the organism, i.e. it does not require any adjustments or offsets. This property allows one to compare the ages of different areas of the human body using the same aging clock. Shortly afterwards, a derivation of Horvath's clock, the IEAA (Intrinsic Epigenetic Age Acceleration), an estimator based on the cellular composition of the blood, was developed.\nA second generation of DNA methylation clocks emerged a few years later and improved on the first in age estimation. This was thanks to the incorporation not only of epigenetic variants such as DNA methylation but also environmental variants such as smoking or chronological age. Among these clocks, the PhenoAge and GrimAge clocks stand out. PhenoAge is an epigenetic clock that takes chronological age into account, and GrimAge uses the mortality risks of age together with the smoking variant among others as a risk factor. Taking into account environmental variants allows GrimAge to outperform any other epigenetic clock in \"predicting death\".\nThird-generation DNA methylation clocks are designed to be applicable across multiple species simultaneously. Specifically, pan-mammalian epigenetic clocks determine the age of tissues from all mammalian species by analyzing cytosine methylation in DNA regions that are highly conserved.\nNew epigenetic clocks based on other markers continue being developed. An aging clock based on nucleosome positioning derived from cell-free DNA was introduced in 2024. In 2025, age-related changes in histone marks have been leveraged to build a new class of epigenetic clocks histone modifications. These new predictors show promise as an alternative to clocks that use cytosine methylation. New age estimation tools are being developed continuously, which also facilitate the prognosis of certain diseases."
