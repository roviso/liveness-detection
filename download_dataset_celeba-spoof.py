# Github repo - https://github.com/Davidzhangyuanhan/CelebA-Spoof
import gdown

# Drive folder with all the zip files - 
# https://drive.google.com/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z
link_list = [
	# 'https://drive.google.com/uc?id=1gd0M4qpxzqrnOkCl7hv7vnCWV4JJTFwM',
	# 'https://drive.google.com/uc?id=1OsOiow42GS4wbE7o1csPA2HtKLSgXT5_',
	# 'https://drive.google.com/uc?id=1_07Q7VA4-4R5fvHpkyoorhD5igeathko',
	# 'https://drive.google.com/uc?id=1Oy-5J10hsrZ2gyHH9U2oet0F92oaEqSq',
	# 'https://drive.google.com/uc?id=1NxlMSgvJSlDJMRFIfq_cT_RRybYS8eNW',
	# 'https://drive.google.com/uc?id=1mpMTb5ODq-9NV4IEVFwEDhUbYSNfVQtC',
	# 'https://drive.google.com/uc?id=14d1TYmcVXg1TlmuywM0HgOKcPqa71Rmq',
	# 'https://drive.google.com/uc?id=1st5Yh8yRQGAmI02iueWPm2rrJN1TpOlC',
	# 'https://drive.google.com/uc?id=1K-6UwtmUvrW7sZqaUvGYIkqpPC1G3K1V',
	# 'https://drive.google.com/uc?id=1wNgEJSBlZuQKePCTbwLM_9jBuIg3Hj1f',
	# 'https://drive.google.com/uc?id=18qd2y1fBPiqQqgSM-vrQq2rMbXeSHWLy',
	# 'https://drive.google.com/uc?id=1HgwT6-NLplcBkowdSHQz5VZEGEsFFDRd',
	# 'https://drive.google.com/uc?id=1-UYx3LoCloTNchmSq5otd4GxkWVTEGs_',
	# 'https://drive.google.com/uc?id=1MiHlWChmvOy53eTyiwWvd6hRV8zSeEYJ',
	# 'https://drive.google.com/uc?id=1WqikxT7XCMZLaPaYfOo0gYZOxeqAiFcP',
	# 'https://drive.google.com/uc?id=1iswrU0QuYSbzwKk0g0TWDD52hbtOps21',
	# 'https://drive.google.com/uc?id=1JPAB1jjLVdn8BBJk4hydDZq-IV5czrGq',
	# 'https://drive.google.com/uc?id=1Qpsou4EJJVd8aVyr9UKdtNvU_Rap3fAU',
	# 'https://drive.google.com/uc?id=1MOfdHWaU_ijqmDjLE34KG5AgmnBuTUnC',
	# 'https://drive.google.com/uc?id=1nKmKMvWV4kqeIPKQkoYnLPGB2QHFS7U1',
	# 'https://drive.google.com/uc?id=1Nkwv2phRRdbEA2Ief1Y9MCrd7UvZIs-G',
	# 'https://drive.google.com/uc?id=1W-IQReuw1Rc6PfTeLpeDxu-1v3n5tRcl',
	# 'https://drive.google.com/uc?id=1lU_RQN_HtB_8yYbRwkCzapfc5-4d0gW1',
	# 'https://drive.google.com/uc?id=1euzrQ4qxGLpOgmW1tIujvDUSdZFlsWmn',
	# 'https://drive.google.com/uc?id=1E6zkuosSUPAQBVGSWg5FlqnhhlACCeYS',
	# 'https://drive.google.com/uc?id=1jZerN3JzvTjxucNnQExqq61gDjxS0l5v',
	# 'https://drive.google.com/uc?id=1F-FI0JO8GLfA2QF5X8SSnjMpNeV5avZH',
	# 'https://drive.google.com/uc?id=1JTGaAL0d6QhwDoY1FRGq_IoexBXKCDAh',
	# 'https://drive.google.com/uc?id=1canVFU2jxJKs6VbcsvHPQeM1qEa8UVwY',
	# 'https://drive.google.com/uc?id=1wt9mOElhnkoYzlVPm5N_69SFhLSvy3YA',
	# 'https://drive.google.com/uc?id=1XwEjdvaBG_xFLVxauZK5MoeeFAnygkYI',
	# 'https://drive.google.com/uc?id=1FhekjS7l-0pDsymm_GM9aMJhh8VwcEah',
	# 'https://drive.google.com/uc?id=1mqXdA81AaAMM-MZvu6h0vCFh0gXWoHNU',
	# 'https://drive.google.com/uc?id=1BQtBSRoA0jFxt7zS9sZgJUFkZUQYj_rU',
	# 'https://drive.google.com/uc?id=1NO1vsdldaVj2pe93QdNfBZ3Ao6VWYvM7',
	# 'https://drive.google.com/uc?id=1xOq1PXanKDpUs1-JAcNaz6Hs5I8MFxxn',
	# 'https://drive.google.com/uc?id=1dihf4baq-_17g2tj1K3ya4Nf7xxMC96O',
	# 'https://drive.google.com/uc?id=1_pA7wv_nMgla0d3KH2cqN8JizL7vZyxP',
	# 'https://drive.google.com/uc?id=1z7c31kyQdrFxbpT6ZblIwAKRnJ9c2MTf',
	# 'https://drive.google.com/uc?id=1oftTxjUVLJdEOGADJzXjVNlUM6_BLlaO',
	# 'https://drive.google.com/uc?id=1jxuF9VFgRwD7ayd3c19cxPb7zLLFM1fp',
	# 'https://drive.google.com/uc?id=16gXdRiWd1tTiuRe5beaVYGXAUhX2m5f_',
	# 'https://drive.google.com/uc?id=1BTtMkRJO-DOhrZhR0kjenegN41mM5qbU',
	# 'https://drive.google.com/uc?id=1JsY9XCjW_Du5VZyPnkuh1Qdajs5aGopt',
	# 'https://drive.google.com/uc?id=11EdKR_DrmAmbvIXhPI6bj7FfMGNh0lAh',
	# 'https://drive.google.com/uc?id=1VX9YmcjCAWXYeQ3OjbEwO7cCPpn0UUzd',
	# 'https://drive.google.com/uc?id=18z-KimWHTh6-T05KzVlENogci5NF-PQH',
	# 'https://drive.google.com/uc?id=1EmVAkbAf3ZfGfVZrZpb7jHqsWqIy3wJa',
	# 'https://drive.google.com/uc?id=1fsS9it_vKV9bgntj8awTx-mRE1uhN9cA',
	# 'https://drive.google.com/uc?id=1lzd-nWagDEqldxkVFRK5-dwGqIf8XYp7',
	# 'https://drive.google.com/uc?id=1krpR7L20uRnOgNo1gx59pOhEcCa_pkyE',
	# 'https://drive.google.com/uc?id=1FZ4OGHJ_CWCf04oysgmCSONRTTYNsf5Y',
	# 'https://drive.google.com/uc?id=1Ac-DvMyCeUzRYsT1R_NrK1zcsQlRy_QL',
	# 'https://drive.google.com/uc?id=1ychf3AazgFb73Z1LIJ_lLhQwvYnOBCXd',
	# 'https://drive.google.com/uc?id=1s11ngviLIOiZcWjX_mkjEdQc7H_oY52-',
	# 'https://drive.google.com/uc?id=1X2Bg5acR0sVkrR4_h3BEMnm6PQFlwzVW',
	# 'https://drive.google.com/uc?id=1g9Ucb8m3l8rV5IpVG1Y2VTkNcZz8DImF',
	# 'https://drive.google.com/uc?id=15n2kpdRqu5rhyIYWIy50JleUHhEXezxt',
	# 'https://drive.google.com/uc?id=1j44juO753d5cp7xLA9ul5bTN8upjgrjY',
	# 'https://drive.google.com/uc?id=1VJoKI6vJJ5lb-JRazmjhDv_NsXzVpQTx',
	# 'https://drive.google.com/uc?id=1G16gR7eM45wAaPt8jVFdDt8Dx510QxNY',
	# 'https://drive.google.com/uc?id=1XZ5oYVAO8J8Zjp_dHb2-4YHWhobE9r2s',
	# 'https://drive.google.com/uc?id=1YI5f9Er6TGq3nlmut2zUfC-bumPtZUOg',
	# 'https://drive.google.com/uc?id=1FO7e7SBv50VOPaQJb_Kh9QfUHXwBrqXb',
	# 'https://drive.google.com/uc?id=1dJ3m7aygdJhcweTmPkZPUxFIYbAM6ysM',
	# 'https://drive.google.com/uc?id=1IjmswQCGQEtpdO3jr9TyJSNj5ikhOCDA',
	# 'https://drive.google.com/uc?id=1P7GD4fTK36dBvASLEVkeG01CUoDbX_Yl',
	# 'https://drive.google.com/uc?id=124cQ_o0MwY3jsieIEHqr2mfHpO3BuKrE',
	# 'https://drive.google.com/uc?id=1eoLXTSOr6MF53D9cElrkg7o5l7yI0xde',
	# 'https://drive.google.com/uc?id=1MZnG9Uzf5q86deEjev4REeOevKsN4eyy',
	# 'https://drive.google.com/uc?id=1U34bffAOUmBsR5YSF_B6Nz2k-97rMFTM',
	# 'https://drive.google.com/uc?id=1v7_ruZ2xvLCl9yQp9dEoPs94qe-rNDjL',
	# 'https://drive.google.com/uc?id=16VogXk2Onsva5i9wszhTL4Sigg-kFlQN',
	'https://drive.google.com/uc?id=1m9dVjwU-KajHbuUMVHgENg62fUdCZg1f',
]

for i, url in enumerate(link_list):
	print((i, url))
	output = 'CelebA_Spoof.zip.' + f'{i+1:03d}'
	gdown.download(url, output, quiet=False)