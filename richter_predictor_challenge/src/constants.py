BINARY_FEATURES = ['has_superstructure_adobe_mud',
                   'has_superstructure_mud_mortar_stone',
                   'has_superstructure_stone_flag',
                   'has_superstructure_cement_mortar_stone',
                   'has_superstructure_mud_mortar_brick',
                   'has_superstructure_cement_mortar_brick',
                   'has_superstructure_timber',
                   'has_superstructure_bamboo',
                   'has_superstructure_rc_non_engineered',
                   'has_superstructure_rc_engineered',
                   'has_superstructure_other',
                   'has_secondary_use',
                   'has_secondary_use_agriculture',
                   'has_secondary_use_hotel',
                   'has_secondary_use_rental',
                   'has_secondary_use_institution',
                   'has_secondary_use_school',
                   'has_secondary_use_industry',
                   'has_secondary_use_health_post',
                   'has_secondary_use_gov_office',
                   'has_secondary_use_use_police',
                   'has_secondary_use_other'
                   ]

CATEGORICAL_FEATURES = [#'count_floors_pre_eq',
                        #'age',
                        # 'area_percentage',
                        # 'height_percentage',
                        'land_surface_condition',
                        'foundation_type',
                        'roof_type',
                        'ground_floor_type',
                        'other_floor_type',
                        'position',
                        'plan_configuration',
                        'legal_ownership_status',
                        #'count_families'
                        ]

HIGH_CARDINAL_FEATURES = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']

NUMERICAL_FEATURES = ['age', 'area_percentage', 'height_percentage', 'count_floors_pre_eq', 'count_families',
                      'area_by_height']