    # #Percentage
    # if configuration == 0: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [True,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [False, 20],
    #         "MinSizeBackground": [False, 1],
    #         "MaxSizeBackground": [False, 20],
    #         "MinSizeShapes": [False, 30],
    #         "MaxSizeShapes": [False, 1],
    #         "Smoothness": [False, 100],
    #     }
    # if configuration == 1: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [True,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [False, 20],
    #         "MinSizeBackground": [True, 1],
    #         "MaxSizeBackground": [True, 20],
    #         "MinSizeShapes": [True, 30],
    #         "MaxSizeShapes": [True, 1],
    #         "Smoothness": [False, 100],
    #     }
    # if configuration == 2: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [True,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [True, 20],
    #         "MinSizeBackground": [True, 1],
    #         "MaxSizeBackground": [True, 20],
    #         "MinSizeShapes": [True, 30],
    #         "MaxSizeShapes": [True, 1],
    #         "Smoothness": [False, 100],
    #     }
    # if configuration == 3: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [True,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [True, 20],
    #         "MinSizeBackground": [True, 1],
    #         "MaxSizeBackground": [True, 20],
    #         "MinSizeShapes": [True, 30],
    #         "MaxSizeShapes": [True, 1],
    #         "Smoothness": [True, 100],
    #     }
    # #BBox
    # if configuration == 4: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[True,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    True], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [False, 20],
    #         "MinSizeBackground": [False, 1],
    #         "MaxSizeBackground": [False, 20],
    #         "MinSizeShapes": [False, 30],
    #         "MaxSizeShapes": [False, 1],
    #         "Smoothness": [False, 100],
    #     }
    # if configuration == 5: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[True,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [False, 20],
    #         "MinSizeBackground": [True, 1],
    #         "MaxSizeBackground": [True, 20],
    #         "MinSizeShapes": [True, 30],
    #         "MaxSizeShapes": [True, 1],
    #         "Smoothness": [False, 100],
    #     }
    # if configuration == 6: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[True,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [True, 20],
    #         "MinSizeBackground": [True, 1],
    #         "MaxSizeBackground": [True, 20],
    #         "MinSizeShapes": [True, 30],
    #         "MaxSizeShapes": [True, 1],
    #         "Smoothness": [False, 100],
    #     }
    # if configuration == 7: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[True,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [True, 20],
    #         "MinSizeBackground": [True, 1],
    #         "MaxSizeBackground": [True, 20],
    #         "MinSizeShapes": [True, 30],
    #         "MaxSizeShapes": [True, 1],
    #         "Smoothness": [True, 100],
    #     }
    # #Scribbles
    # if configuration == 8: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [True, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [False, 20],
    #         "MinSizeBackground": [False, 1],
    #         "MaxSizeBackground": [False, 20],
    #         "MinSizeShapes": [False, 30],
    #         "MaxSizeShapes": [False, 1],
    #         "Smoothness": [False, 100],
    #     }
    # if configuration == 9: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [True, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [False, 20],
    #         "MinSizeBackground": [True, 1],
    #         "MaxSizeBackground": [True, 20],
    #         "MinSizeShapes": [True, 30],
    #         "MaxSizeShapes": [True, 1],
    #         "Smoothness": [False, 100],
    #     }
    # if configuration == 10: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [True, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [True, 20],
    #         "MinSizeBackground": [True, 1],
    #         "MaxSizeBackground": [True, 20],
    #         "MinSizeShapes": [True, 30],
    #         "MaxSizeShapes": [True, 1],
    #         "Smoothness": [False, 100],
    #     }
    # if configuration == 11: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [True, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [True, 20],
    #         "MinSizeBackground": [True, 1],
    #         "MaxSizeBackground": [True, 20],
    #         "MinSizeShapes": [True, 30],
    #         "MaxSizeShapes": [True, 1],
    #         "Smoothness": [True, 100],
    #     }
    #Region
    # if configuration == 0: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [True,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [False, 20],
    #         "MinSizeBackground": [False, 1],
    #         "MaxSizeBackground": [False, 20],
    #         "MinSizeShapes": [False, 30],
    #         "MaxSizeShapes": [False, 1],
    #         "Smoothness": [False, 100],
    #     }
    # if configuration == 1: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [True,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [False, 20],
    #         "MinSizeBackground": [True, 1],
    #         "MaxSizeBackground": [True, 20],
    #         "MinSizeShapes": [True, 30],
    #         "MaxSizeShapes": [True, 1],
    #         "Smoothness": [False, 100],
    #     }
    # if configuration == 2: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [True,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [True, 20],
    #         "MinSizeBackground": [True, 1],
    #         "MaxSizeBackground": [True, 20],
    #         "MinSizeShapes": [True, 30],
    #         "MaxSizeShapes": [True, 1],
    #         "Smoothness": [False, 100],
    #     }
    # if configuration == 3: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [True,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [True, 20],
    #         "MinSizeBackground": [True, 1],
    #         "MaxSizeBackground": [True, 20],
    #         "MinSizeShapes": [True, 30],
    #         "MaxSizeShapes": [True, 1],
    #         "Smoothness": [True, 100],
    #     }
    # #Point
    # if configuration == 4: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    True], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [True, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [False, 20],
    #         "MinSizeBackground": [False, 1],
    #         "MaxSizeBackground": [False, 20],
    #         "MinSizeShapes": [False, 30],
    #         "MaxSizeShapes": [False, 1],
    #         "Smoothness": [False, 100],
    #     }
    # if configuration == 5: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [True, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [False, 20],
    #         "MinSizeBackground": [True, 1],
    #         "MaxSizeBackground": [True, 20],
    #         "MinSizeShapes": [True, 30],
    #         "MaxSizeShapes": [True, 1],
    #         "Smoothness": [False, 100],
    #     }
    # if configuration == 6: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [True, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [True, 20],
    #         "MinSizeBackground": [True, 1],
    #         "MaxSizeBackground": [True, 20],
    #         "MinSizeShapes": [True, 30],
    #         "MaxSizeShapes": [True, 1],
    #         "Smoothness": [False, 100],
    #     }
    # if configuration == 7: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [True, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [False,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [True, 20],
    #         "MinSizeBackground": [True, 1],
    #         "MaxSizeBackground": [True, 20],
    #         "MinSizeShapes": [True, 30],
    #         "MaxSizeShapes": [True, 1],
    #         "Smoothness": [True, 100],
    #     }
    # #Adjacency
    # if configuration == 8: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [True,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [False, 20],
    #         "MinSizeBackground": [False, 1],
    #         "MaxSizeBackground": [False, 20],
    #         "MinSizeShapes": [False, 30],
    #         "MaxSizeShapes": [False, 1],
    #         "Smoothness": [False, 100],
    #     }
    # if configuration == 9: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [True,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [False, 20],
    #         "MinSizeBackground": [True, 1],
    #         "MaxSizeBackground": [True, 20],
    #         "MinSizeShapes": [True, 30],
    #         "MaxSizeShapes": [True, 1],
    #         "Smoothness": [False, 100],
    #     }
    # if configuration == 10: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [True,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [True, 20],
    #         "MinSizeBackground": [True, 1],
    #         "MaxSizeBackground": [True, 20],
    #         "MinSizeShapes": [True, 30],
    #         "MaxSizeShapes": [True, 1],
    #         "Smoothness": [False, 100],
    #     }
    # if configuration == 11: 
    #     configuration_dict = {
    #         "seed": 42,
    #         "FullySupervisedPercentage": 0,

    #         "logicLossMultiplier": 0.01,
    #         #                   useTruePercentages,  useAtleast
    #         "ImageLevel": [False,       True     ,       False   , 2], #note background percentage is ignored

    #         #               useTruePercentages      outsideBbox  BboxAtmost   linear-or-prob   useNew  useOnlyNew
    #         "BBox": [[False,      False       , 1],   [True, 0.2],  [False, 1],      "linear", True,    False], 
    #         "BBoxFull": [False, 1,"linear"], #linear or prob
    #         "Scribbles": [False, 1],

    #         #              useTruePercentages, generalfactor, boost factor for atleast minsize in area
    #         "Area": [False,       False         ,     1,                   10,                               False], 
    #         "Point": [False, 10],
    #         #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
    #         "Adjacency": [True,  True,         30,              0.0001,                 False,                  False],
    #         #                 norm-mult   impl   impl-mult   symmetric 
    #         "Relations": [False,   2,      True,    0.1,       True],
    #         "SoftRelations": [False, 1],
    #         #global constraints:
    #         "OneHot": [True, 20],
    #         "MinSizeBackground": [True, 1],
    #         "MaxSizeBackground": [True, 20],
    #         "MinSizeShapes": [True, 30],
    #         "MaxSizeShapes": [True, 1],
    #         "Smoothness": [True, 100],
    #     }