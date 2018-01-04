import scrapy
from scrapy import Request


class song_infoSpider(scrapy.Spider):
    name = 'song_info'
    allowed_domains = ['jazz-on-line.com']
    start_urls = ['http://www.jazz-on-line.com/pageinterrogation.php']



    def parse(self, response):
        song_list = response.xpath('//tr')

        for i, song in enumerate(song_list):
            if i > 17 :
                artist = song.xpath('td[2]/font/text()').extract_first()
                title = song.xpath('td[3]/font/a/text()').extract()
                url_sample = song.xpath('td[3]/font/a/@href').extract()
                info = song.xpath('td[3]/font/text()').extract()
                url_download = song.xpath('td[6]/font/a/@href').extract()

                yield { 'Title' : title,
                        'URL': url_sample,
                        'Artist': artist,
                        'Info' : info,
                        'Download_link': url_download
                        }
