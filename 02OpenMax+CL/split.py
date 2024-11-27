train_splits = {
    'radar': [
        [1, 2, 4, 5],
        [0, 2, 3, 4],
        [1, 2, 4, 6],
        [0, 1, 3, 5],
        [1, 4, 5, 6]
    ],
    'ADSB': [
        [4, 7, 10, 15, 17, 22, 27, 29, 30, 32, 33, 35, 40, 43, 45],
        [2, 3, 5, 6, 8, 9, 11, 27, 33, 35, 36, 38, 39, 42, 48],
        [0, 1, 4, 6, 10, 18, 20, 24, 26, 27, 30, 31, 32, 34, 40],
        [4, 5, 7, 8, 11, 16, 18, 29, 30, 31, 32, 33, 39, 40, 45],
        [3, 4, 12, 14, 15, 16, 17, 19, 24, 27, 35, 36, 40, 41, 45]
    ],
}

test_splits = {
    'radar':[
        [1, 2, 4, 5, 0, 3, 6],
        [0, 2, 3, 4, 1, 5, 6],
        [1, 2, 4, 6, 0, 3, 5],
        [0, 1, 3, 5, 2, 4, 6],
        [1, 4, 5, 6, 0, 2, 3],

        # [1, 2, 4, 5, 0, 3],
        # [0, 2, 3, 4, 1, 5],
        # [1, 2, 4, 6, 0, 3],
        # [0, 1, 3, 5, 2, 4],
        # [1, 4, 5, 6, 0, 2],
        #
        # [1, 2, 4, 5, 0],
        # [0, 2, 3, 4, 1],
        # [1, 2, 4, 6, 0],
        # [0, 1, 3, 5, 2],
        # [1, 4, 5, 6, 0],
        #
        # [1, 2, 4, 5],
        # [0, 2, 3, 4],
        # [1, 2, 4, 6],
        # [0, 1, 3, 5],
        # [1, 4, 5, 6]
    ],
    'ADSB': [
        [4, 7, 10, 15, 17, 22, 27, 29, 30, 32, 33, 35, 40, 43, 45, 0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14, 16, 18, 19,
         20, 21, 23, 24, 25, 26, 28, 31, 34, 36, 37, 38, 39, 41, 42, 44, 46, 47, 48],
        [2, 3, 5, 6, 8, 9, 11, 27, 33, 35, 36, 38, 39, 42, 48, 0, 1, 4, 7, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
         22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 34, 37, 40, 41, 43, 44, 45, 46, 47],
        [0, 1, 4, 6, 10, 18, 20, 24, 26, 27, 30, 31, 32, 34, 40, 2, 3, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 21,
         22, 23, 25, 28, 29, 33, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48],
        [4, 5, 7, 8, 11, 16, 18, 29, 30, 31, 32, 33, 39, 40, 45, 0, 1, 2, 3, 6, 9, 10, 12, 13, 14, 15, 17, 19, 20, 21,
         22, 23, 24, 25, 26, 27, 28, 34, 35, 36, 37, 38, 41, 42, 43, 44, 46, 47, 48],
        [3, 4, 12, 14, 15, 16, 17, 19, 24, 27, 35, 36, 40, 41, 45, 0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 13, 18, 20, 21, 22,
         23, 25, 26, 28, 29, 30, 31, 32, 33, 34, 37, 38, 39, 42, 43, 44, 46, 47, 48],

        # [4, 7, 10, 15, 17, 22, 27, 29, 30, 32, 33, 35, 40, 43, 45, 0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14, 16, 18, 19,
        #  20, 21, 23, 24, 25, 26, 28, 31, 34, 36, 37, 38, 39, 41, 42],
        # [2, 3, 5, 6, 8, 9, 11, 27, 33, 35, 36, 38, 39, 42, 48, 0, 1, 4, 7, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        #  22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 34, 37, 40, 41, 43],
        # [0, 1, 4, 6, 10, 18, 20, 24, 26, 27, 30, 31, 32, 34, 40, 2, 3, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 21,
        #  22, 23, 25, 28, 29, 33, 35, 36, 37, 38, 39, 41, 42, 43, 44],
        # [4, 5, 7, 8, 11, 16, 18, 29, 30, 31, 32, 33, 39, 40, 45, 0, 1, 2, 3, 6, 9, 10, 12, 13, 14, 15, 17, 19, 20, 21,
        #  22, 23, 24, 25, 26, 27, 28, 34, 35, 36, 37, 38, 41, 42, 43],
        # [3, 4, 12, 14, 15, 16, 17, 19, 24, 27, 35, 36, 40, 41, 45, 0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 13, 18, 20, 21, 22,
        #  23, 25, 26, 28, 29, 30, 31, 32, 33, 34, 37, 38, 39, 42, 43],
        #
        # [4, 7, 10, 15, 17, 22, 27, 29, 30, 32, 33, 35, 40, 43, 45, 0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14, 16, 18, 19,
        #  20, 21, 23, 24, 25, 26, 28, 31, 34, 36],
        # [2, 3, 5, 6, 8, 9, 11, 27, 33, 35, 36, 38, 39, 42, 48, 0, 1, 4, 7, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        #  22, 23, 24, 25, 26, 28, 29, 30, 31, 32],
        # [0, 1, 4, 6, 10, 18, 20, 24, 26, 27, 30, 31, 32, 34, 40, 2, 3, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 21,
        #  22, 23, 25, 28, 29, 33, 35, 36, 37, 38],
        # [4, 5, 7, 8, 11, 16, 18, 29, 30, 31, 32, 33, 39, 40, 45, 0, 1, 2, 3, 6, 9, 10, 12, 13, 14, 15, 17, 19, 20, 21,
        #  22, 23, 24, 25, 26, 27, 28, 34, 35, 36],
        # [3, 4, 12, 14, 15, 16, 17, 19, 24, 27, 35, 36, 40, 41, 45, 0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 13, 18, 20, 21, 22,
        #  23, 25, 26, 28, 29, 30, 31, 32, 33, 34],
        #
        # [4, 7, 10, 15, 17, 22, 27, 29, 30, 32, 33, 35, 40, 43, 45, 0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14, 16, 18, 19,
        #  20, 21, 23, 24, 25],
        # [2, 3, 5, 6, 8, 9, 11, 27, 33, 35, 36, 38, 39, 42, 48, 0, 1, 4, 7, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        #  22, 23, 24, 25, 26],
        # [0, 1, 4, 6, 10, 18, 20, 24, 26, 27, 30, 31, 32, 34, 40, 2, 3, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 21,
        #  22, 23, 25, 28, 29],
        # [4, 5, 7, 8, 11, 16, 18, 29, 30, 31, 32, 33, 39, 40, 45, 0, 1, 2, 3, 6, 9, 10, 12, 13, 14, 15, 17, 19, 20, 21,
        #  22, 23, 24, 25, 26],
        # [3, 4, 12, 14, 15, 16, 17, 19, 24, 27, 35, 36, 40, 41, 45, 0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 13, 18, 20, 21, 22,
        #  23, 25, 26, 28, 29],
        #
        # [4, 7, 10, 15, 17, 22, 27, 29, 30, 32, 33, 35, 40, 43, 45, 0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14, 16, 18, 19],
        # [2, 3, 5, 6, 8, 9, 11, 27, 33, 35, 36, 38, 39, 42, 48, 0, 1, 4, 7, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        # [0, 1, 4, 6, 10, 18, 20, 24, 26, 27, 30, 31, 32, 34, 40, 2, 3, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 21],
        # [4, 5, 7, 8, 11, 16, 18, 29, 30, 31, 32, 33, 39, 40, 45, 0, 1, 2, 3, 6, 9, 10, 12, 13, 14, 15, 17, 19, 20, 21],
        # [3, 4, 12, 14, 15, 16, 17, 19, 24, 27, 35, 36, 40, 41, 45, 0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 13, 18, 20, 21, 22],
        #
        # [4, 7, 10, 15, 17, 22, 27, 29, 30, 32, 33, 35, 40, 43, 45, 0, 1, 2, 3, 5, 6, 8, 9, 11, 12],
        # [2, 3, 5, 6, 8, 9, 11, 27, 33, 35, 36, 38, 39, 42, 48, 0, 1, 4, 7, 10, 12, 13, 14, 15, 16],
        # [0, 1, 4, 6, 10, 18, 20, 24, 26, 27, 30, 31, 32, 34, 40, 2, 3, 5, 7, 8, 9, 11, 12, 13, 14],
        # [4, 5, 7, 8, 11, 16, 18, 29, 30, 31, 32, 33, 39, 40, 45, 0, 1, 2, 3, 6, 9, 10, 12, 13, 14],
        # [3, 4, 12, 14, 15, 16, 17, 19, 24, 27, 35, 36, 40, 41, 45, 0, 1, 2, 5, 6, 7, 8, 9, 10, 11],
        #
        # [4, 7, 10, 15, 17, 22, 27, 29, 30, 32, 33, 35, 40, 43, 45, 0, 1, 2, 3, 5],
        # [2, 3, 5, 6, 8, 9, 11, 27, 33, 35, 36, 38, 39, 42, 48, 0, 1, 4, 7, 10],
        # [0, 1, 4, 6, 10, 18, 20, 24, 26, 27, 30, 31, 32, 34, 40, 2, 3, 5, 7, 8],
        # [4, 5, 7, 8, 11, 16, 18, 29, 30, 31, 32, 33, 39, 40, 45, 0, 1, 2, 3, 6],
        # [3, 4, 12, 14, 15, 16, 17, 19, 24, 27, 35, 36, 40, 41, 45, 0, 1, 2, 5, 6],

        [4, 7, 10, 15, 17, 22, 27, 29, 30, 32, 33, 35, 40, 43, 45],
        [2, 3, 5, 6, 8, 9, 11, 27, 33, 35, 36, 38, 39, 42, 48],
        [0, 1, 4, 6, 10, 18, 20, 24, 26, 27, 30, 31, 32, 34, 40],
        [4, 5, 7, 8, 11, 16, 18, 29, 30, 31, 32, 33, 39, 40, 45],
        [3, 4, 12, 14, 15, 16, 17, 19, 24, 27, 35, 36, 40, 41, 45]
    ]
}