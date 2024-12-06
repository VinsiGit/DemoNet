adresses = ['Noltheniusstraat 29, 3533 SG  UTRECHT', 'Von Weberstraat 29, 3533 EC  UTRECHT', 'Chopinstraat 42, 3533 EN  UTRECHT', 'Johann Sebastian Bachstr 55, 3533 XB  UTRECHT', 'Kweekhoeve 22, 3438 MB  NIEUWEGEIN', 'Nederhoeve 12, 3438 LH  NIEUWEGEIN', 'Ina Boudier-Bakkerhove 12, 3438 PA  NIEUWEGEIN', 'Bertus Aafjeshove 32, 3437 JN  NIEUWEGEIN', 'Geraniumstraat 21, 3551 HA  UTRECHT', 'Robijnlaan 27, 3523 BV  UTRECHT', 'Topaaslaan 16, 3523 AT  UTRECHT', 'Rijnlaan 214, 3522 BX  UTRECHT', 'Diezestraat 14, 3522 GZ  UTRECHT', 'Maasstraat 20, 3522 TH  UTRECHT', 'Runstraat 8, 3522 RH  UTRECHT', 'Noordzeestraat 41, 3522 PJ  UTRECHT', 'Croesestraat 61, 3522 AC  UTRECHT', 'Abstederdijk 239, 3582 BJ  UTRECHT', 'Piet Heinstraat 15, 3582 BX  UTRECHT', 'Aurorastraat 14, 3581 LV  UTRECHT', 'Eikstraat 21, 3581 XJ  UTRECHT', 'Beukstraat 22, 3581 XG  UTRECHT', 'Prinsenstraat 13, 3581 JR  UTRECHT', 'Poortstraat 21, 3572 HB  UTRECHT', 'Poortstraat 60, 3572 HL  UTRECHT', 'van Doesburglaan 2, 3431 GB  NIEUWEGEIN', 'Rietveldlaan 8, 3431 GD  NIEUWEGEIN', 'Mondriaanlaan 10, 3431 GA  NIEUWEGEIN', 'Kerkveld 67, 3431 EC  NIEUWEGEIN', 'Woudenbergseweg 3, 3701 BA  ZEIST', 'Krullelaan 28, 3701 TD  ZEIST', 'Wilhelminalaan 30, 3701 BL  ZEIST', 'Choisyweg 5, 3701 TA  ZEIST', 'De Vuursche 1, 3452 JT  VLEUTEN', 'De Vuursche 16, 3452 JT  VLEUTEN', 'De Vuursche 46, 3452 JT  VLEUTEN', 'De Vuursche 72, 3452 JT  VLEUTEN', 'Utrechtse Heuvelrug 68, 3452 JA  VLEUTEN', 'Zilverschoonlaan 79, 3452 AA  VLEUTEN', 'Schoolstraat 29, 3451 AA  VLEUTEN']
years = ['1954', '1955', '1935', '1919', '2019', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
surfaces = ['126', '115', '110', '120', '276', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
BAGnum = ['0344100000081447', '0344100000067920', '0344100000050569', '0344100000039962', '0356100000125385', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

print('[')
for i, adress in enumerate(adresses):
    print('\t{'+
    f'''
        "image_id": {i+1},
        "image_path": "/images/image{i+1}.png",
        "address": "{adress}",
        "year": {years[i]},
        "area": {surfaces[i]},
        "BAG": "{BAGnum[i]}"
    '''+'\t},')
print(']')