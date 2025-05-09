class Config:
    """
    Configuration file for the analysis.
    """
    def __init__(self):
        self.seed = 1234

        # Folders
        self.input_folder = "../data/raw"
        self.data_folder = "../data/processed"
        self.output_folder = "../outputs"

        ### Study area polygons and merging of tiled external LUC products
        # Input (GADM boundaries)
        self.vnm_gadm_path = f"{self.input_folder}/boundaries/gadm41_VNM_0.json"
        self.vnm_provinces_gadm_path = f"{self.input_folder}/boundaries/gadm41_VNM_1.json"
        self.lao_gadm_path = f"{self.input_folder}/boundaries/gadm41_LAO_0.json"
        self.lao_provinces_gadm_path = f"{self.input_folder}/boundaries/gadm41_LAO_1.json"
        self.khm_gadm_path = f"{self.input_folder}/boundaries/gadm41_KHM_0.json"
        self.chn_gadm_path = f"{self.input_folder}/boundaries/gadm41_CHN_0.json"
        self.tha_gadm_path = f"{self.input_folder}/boundaries/gadm41_THA_0.json"
        self.mmr_gadm_path = f"{self.input_folder}/boundaries/gadm41_MMR_0.json"

        # Input (GLC_FCS30D data)
        self.luc_fcs30_folder = f"{self.input_folder}/luc/GLC_FCS30D_19852022maps_E100-E105"
        self.luc_fcs30_1985_path = f"{self.input_folder}/luc/luc_glc_fcs30_1985.tif" 
        self.luc_fcs30_1990_path = f"{self.input_folder}/luc/luc_glc_fcs30_1990.tif"
        self.luc_fcs30_legend_path = f"{self.input_folder}/luc/legend_fcs30.csv"
        
        # Input (GMW data and mangrove study area)
        self.vnm_eez_land_path = f"{self.input_folder}/boundaries/EEZ_land_union_v4_202410/EEZ_land_union_v4_202410.shp"
        self.gmw_1996_folder = f"{self.input_folder}/luc/gmw_v3_1996"
        self.gmw_1996_merged_path = f"{self.input_folder}/luc/gmw_1996_merged.tif"

        # Processing
        self.study_area_path = f"{self.data_folder}/study_areas/study_area.geojson"
        self.study_area_lao_path = f"{self.data_folder}/study_areas/study_area_lao.geojson"
        self.study_area_vnm_path = f"{self.data_folder}/study_areas/study_area_vnm.geojson"
        self.study_area_svnm_path = f"{self.data_folder}/study_areas/study_area_svnm.geojson"
        self.study_area_nvnm_path = f"{self.data_folder}/study_areas/study_area_nvnm.geojson"

        self.cropping_geoms = {
            "lao": f"{self.data_folder}/study_areas/gadm_lao.geojson",
            "vnm": f"{self.data_folder}/study_areas/gadm_vnm.geojson",
            "svnm": f"{self.data_folder}/study_areas/gadm_svnm.geojson",
            "nvnm": f"{self.data_folder}/study_areas/gadm_nvnm.geojson"
        }

        ### Preprocessing 
        # Input
        # Map sheet index
        self.map_sheet_folder = "../../topo-maps-paper/data/raw/map_sheets" #f"{self.input_folder}/map_sheets"
        self.map_sheet_index_path = f"{self.input_folder}/map_sheet_index.csv" 
        self.map_sheet_index_geo_path = f"{self.input_folder}/map_sheet_index_geo.geojson" 
        self.map_sheet_index_geo_overedge_path = f"{self.input_folder}/map_sheet_index_geo_overedge.geojson" 

        # Additional training samples
        self.adaptive_samples_batch2 = f"{self.input_folder}/adaptive_tile_samples/batch2.csv"
        self.adaptive_samples_batch3 = f"{self.input_folder}/adaptive_tile_samples/batch3.csv"

        # Processing
        self.tile_catalog_path = f"{self.data_folder}/tile_catalog.geojson"
        self.sample_catalog_path = f"{self.data_folder}/sample_catalog.geojson"
        self.tile_folder = f"{self.data_folder}/image_tiles"

        # Parameters
        self.map_sheet_index_crs = "EPSG:4131"
        self.tile_size = 256
        self.samples_per_stratum = 80

        ### Training data creation
        # Inputs
        self.label_folder = f"{self.input_folder}/label_tiles"
        self.topo_legend_path = f"{self.input_folder}/legend_topo.csv"

        # Processing
        self.training_data_path =  f"{self.data_folder}/training_data.npz"

        # Parameters
        self.prop_train = 0.75

        ### Model Training
        # Processing
        self.checkpoint_folder = f"{self.output_folder}/model_ckpts"
        self.eval_folder = f"{self.output_folder}/model_eval"
        self.wandb_project = "your_project_name"

        # Parameters
        self.classes = 17
        self.encoder_weights = "imagenet"
        self.ignore_index = 0 # index to ignore during model training
        self.batch_size = 8
        self.lr = 1e-4
        self.max_epochs = 2000
        self.check_val_every_n_epoch = 1

        # Augmentation parameters
        self.rot_deg = 15
        self.aug_settings = {
            "no_color_jitter": {"brightness": 0, "contrast": 0, "saturation": 0, "hue": 0},
            "low_color_jitter": {"brightness": 0.05, "contrast": 0.05, "saturation": 0.05, "hue": 0.025},
            "high_color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1}
        }

        # Expected class counts for class weight calculation
        self.expected_class_counts = [
            7287034,
            144067358,
            103606797,
            315206949,
            2925708224,
            20929206571,
            3484907717,
            135521207,
            353009170,
            158330977,
            221475732,
            2555252089,
            1384970700,
            99455015,
            28412197,
            540679935
            ]

        ### Map sheet processing and reprojection
        self.map_sheet_processed_folder = f"{self.data_folder}/map_sheets"

        # Parameters
        self.nodata = 0
        self.compress = "LZW"
        self.output_crs = "ESRI:102028"
        self.res_detailed = 4
        self.res_coarse = 30
        self.buffer_map_sheet_extent = 0.00004 # buffer to avoid gaps between map sheets
    
        ### Model prediction 
        # Inputs
        self.final_model_path = f"{self.checkpoint_folder}/unet++_resnet50_focal_walpha_0.0_aug_no_color_jitter"

        # Parameters
        self.overlap = 64 # overlap for shifting window during prediction

        ### Postprocessing
        # Inputs
        self.postprocessing_input_folder = f"{self.input_folder}/postprocessing"
        self.polygons_imputation_water = f"{self.postprocessing_input_folder}/polygons_inmap_legends.geojson"
        self.polygons_imputation_forest = f"{self.postprocessing_input_folder}/polygons_forest.geojson"
        self.polygons_imputation_other = f"{self.postprocessing_input_folder}/polygons_impute.geojson"
        self.polygons_dmz = f"{self.postprocessing_input_folder}/polygons_dmz.geojson"

        # Parameters
        self.min_area_impute = 20
        self.keys_brushwood_to_forest = [59512, 59522]

        # Outputs
        self.map_sheets_merged_path = f"{self.map_sheet_processed_folder}/crop.tif"
        self.raster_topo_l2_path = f"{self.map_sheet_processed_folder}/post2.tif"
        self.topo_eval_lao = f"{self.map_sheet_processed_folder}/post2_lao.tif"
        self.topo_eval_vnm = f"{self.map_sheet_processed_folder}/post3_vnm.tif"

        ### Accuracy evaluation
        self.acc_eval_folder = f"{self.data_folder}/acc_eval"

        # Inputs
        # Initial LUC predictions used for stratification of test set
        self.topo_sampling_lao = f"{self.input_folder}/post2_lao_stratum.tif"
        self.topo_sampling_vnm = f"{self.input_folder}/post3_vnm_stratum.tif"
        # Test labels after manual labelling of test sample locations
        self.test_labels_vnm = f"{self.input_folder}/test_labels_vnm.csv"
        self.test_labels_lao = f"{self.input_folder}/test_labels_lao.csv"

        # Processing
        self.class_counts_eval_vnm = f"{self.acc_eval_folder}/class_counts_vnm.csv"
        self.class_counts_eval_lao = f"{self.acc_eval_folder}/class_counts_lao.csv"
        self.strata_counts_eval_vnm = f"{self.acc_eval_folder}/strata_counts_vnm.csv"
        self.strata_counts_eval_lao = f"{self.acc_eval_folder}/strata_counts_lao.csv"

        self.test_samples_init_vnm = f"{self.acc_eval_folder}/test_samples_init_vnm.csv"
        self.test_samples_init_lao = f"{self.acc_eval_folder}/test_samples_init_lao.csv"
        self.test_samples_vnm = f"{self.acc_eval_folder}/test_samples_vnm.csv"
        self.test_samples_lao = f"{self.acc_eval_folder}/test_samples_lao.csv"
        self.test_samples_geo_vnm = f"{self.acc_eval_folder}/test_samples_vnm.geojson"
        self.test_samples_geo_lao = f"{self.acc_eval_folder}/test_samples_lao.geojson"
        self.test_samples_folder_vnm = f"{self.acc_eval_folder}/test_samples_vnm"
        self.test_samples_folder_lao = f"{self.acc_eval_folder}/test_samples_lao"

        self.test_labels_pred_vnm = f"{self.acc_eval_folder}/test_labels_pred_vnm.csv"
        self.test_labels_pred_lao = f"{self.acc_eval_folder}/test_labels_pred_lao.csv"

        # Parameters
        self.n_test_samples_init_per_class = 1000
        self.n_test_samples_sel_per_class = 100
        self.n_test_samples_sel_by_prop = 500

        # Outputs
        self.confusion_matrix_vnm = f"{self.output_folder}/confusion_matrix_vnm.csv"
        self.confusion_matrix_lao = f"{self.output_folder}/confusion_matrix_lao.csv"

        ### Analysis

        ## Forest cover
        self.analysis_forest_folder = f"{self.data_folder}/analysis/forest"

        # Processing
        self.raster_fcs30_1990_proj_path = f"{self.analysis_forest_folder}/fcs30_1990_proj.tif"
        self.raster_topo_aligned_fcs30_path = f"{self.analysis_forest_folder}/topo_aligned_fcs30.tif"
        self.raster_forest_loss_path = f"{self.analysis_forest_folder}/forest_loss_topo_fcs30_1990.tif"
        self.forest_stats_map_sheets = f"{self.analysis_forest_folder}/forest_stats_map_sheets.geojson"

        # Parameters
        self.fcs30_resolution = 30
        self.topo_forest_vals = [6]
        self.fcs30_forest_vals = [51, 52, 61, 62, 71, 72, 81, 82, 91, 92]

        # Outputs
        self.forest_stats_study_area = f"{self.output_folder}/forest_change_stats.csv"

        ## Mangroves
        self.analysis_mangrove_folder = f"{self.data_folder}/analysis/mangrove"

        # Inputs
        self.u_minh_path = f"{self.input_folder}/boundaries/u_minh.geojson"
        
        # Processing
        self.gmw_1996_proj_path = f"{self.analysis_mangrove_folder}/gmw_1996_proj.tif"
        self.topo_aligned_gmw_path = f"{self.analysis_mangrove_folder}//topo_aligned_gmw.tif"
        self.study_areas_mangrove_path = f"{self.analysis_mangrove_folder}/study_areas_mangrove.geojson"
        self.mangrove_stats_map_sheets = f"{self.analysis_mangrove_folder}/mangrove_stats_map_sheets.geojson"
        self.mangrove_topo_vector = f"{self.analysis_mangrove_folder}/mangrove_topo.geojson"
        self.mangrove_gmw_vector = f"{self.analysis_mangrove_folder}/mangrove_gmw.geojson"
        self.mangrove_gains_path = f"{self.analysis_mangrove_folder}/mangrove_gains.geojson"
        self.mangrove_losses_path = f"{self.analysis_mangrove_folder}/mangrove_losses.geojson"
        self.mangrove_stable_path = f"{self.analysis_mangrove_folder}/mangrove_stable.geojson"

        # Parameters
        self.gmw_resolution = 25
        self.mangrove_raster_idx = 11
        
        # Outputs
        self.mangrove_stats_study_area = f"{self.output_folder}/mangrove_change_stats.csv" 

        ### Figures  
        # Inputs
        self.example_map_sheet_path = f"{self.input_folder}/VN_Sai Gon_L7014_6330-4_50000_1970_Ed_4.tif"
        self.topo_plot_lao = f"{self.map_sheet_processed_folder}/post2_lao_30m.tif"
        self.topo_plot_vnm = f"{self.map_sheet_processed_folder}/post3_vnm_30m.tif"

        # Outputs
        self.plot_folder = f"{self.output_folder}/plots"