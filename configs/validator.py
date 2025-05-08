import yaml

class Validator():
    type_mapping = {
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list
    }

    @classmethod
    def validate_cfg(cls, cfg_instance, cfg_dict):
        for name, parameter in cfg_dict.items():
            value = parameter['value']
            expected_type = cls.type_mapping[parameter['type']]
            options = parameter.get('options', None)

            if value is not None and not isinstance(value, expected_type):
                raise TypeError(
                    f"Expected type `{expected_type.__name__}` for parameter `{name}`, "
                    f"but got type `{type(value).__name__}`."
                    )
            
            attr_name = f'_validate_{name.lower()}'
            if hasattr(cls, attr_name):
                getattr(cls, attr_name)(value, options, cfg_instance)

    @classmethod
    def _validate_model(cls, value, options, instance):
        if not value in options:
            raise ValueError(f"Value of `MODEL` must be one of {options}.")                
    
    @classmethod
    def _validate_train_set_composition(cls, value, options, instance):
        if not value in options:
            raise ValueError(f"Value of `TRAIN_SET_COMPOSITION` must be one of {options}.")
        
        if instance.MODEL in ['AuxNet-C', 'ClsCNN'] and value == 'boundary':
            raise ValueError("Value conflict between `MODEL` and `TRAIN_SET_COMPOSITION`! "
                            f"Model `{instance.MODEL}` cannot be trained with `boundary` dataset.")

    @classmethod   
    def _validate_test_set_composition(cls, value, options, instance):
        if not value in options:
            raise ValueError(f"Value of `TEST_SET_COMPOSITION` must be one of {options}.")
        
        if instance.MODEL in ['AuxNet-C', 'ClsCNN'] and value == 'boundary':
            raise ValueError("Value conflict between `MODEL` and `TEST_SET_COMPOSITION`! "
                            f"Model `{instance.MODEL}` cannot be tested on `boundary` dataset.")

    @classmethod 
    def _validate_dataset_dir(cls, value, options, instance):
        pass

    @classmethod
    def _validate_output_dir(cls, value, options, instance):
        pass
    
    @classmethod
    def _validate_input_image_size(cls, value, options, instance):
        if not value > 0:
            raise ValueError("Value of `INPUT_IMAGE_SIZE` must be greater than 0.")
        
        if not value % 2 == 0:
            raise ValueError("Value of `INPUT_IMAGE_SIZE` must be an even number.")

        if not value/(2 ** (instance.NUM_LAYERS - 1)) >= 4:
            raise ValueError("Value conflict between `INPUT_IMAGE_SIZE` and `NUM_LAYERS`! "
                            "INPUT_IMAGE_SIZE/2^(NUM_LAYERS - 1) must be greater than or equal to 4.")

    @classmethod   
    def _validate_num_channels(cls, value, options, instance):
        if not value > 0:
            raise ValueError("Value of `NUM_CHANNELS` must be greater than 0.")
    
    @classmethod
    def _validate_sdm_kernel_size(cls, value, options, instance):
        if not value > 0:
            raise ValueError("Value of `SDM_KERNEL` must be greater than 0.")

        if not value % 2 == 1:
            raise ValueError("Value of `SDM_KERNEL` must be an odd number.")

    @classmethod 
    def _validate_sdm_clamp_delta(cls, value, options, instance):
        if not 0 < value < 1:
            raise ValueError("Value of `CLAMP_DELTA` must be between 0 and 1.")
    
    @classmethod
    def _validate_init_lr(cls, value, options, instance):
        if not 0 < value < 1:
            raise ValueError("Value of `INIT_LR` must be between 0 and 1.")

        if not value < instance.BASE_LR:
            raise ValueError("Value conflict between `INIT_LR` and `BASE_LR`! "
                            "`INIT_LR` must be less than `BASE_LR`.")
    
    @classmethod
    def _validate_base_lr(cls, value, options, instance):
        if not 0 < value < 1:
            raise ValueError("Value of `BASE_LR` must be between 0 and 1.")
        
        if not value > instance.INIT_LR:
            raise ValueError("Value conflict between `BASE_LR` and `INIT_LR`! "
                            "`BASE_LR` must be greater than `INIT_LR`.")
    
    @classmethod
    def _validate_l2_decay(cls, value, options, instance):
        if not 0 < value < 1:
            raise ValueError("Value of `L2_DECAY` must be between 0 and 1.")
    
    @classmethod
    def _validate_warmup_epochs(cls, value, options, instance):
        if not value >= 0:
            raise ValueError("Value of `WARMUP_EPOCHS` cannot be negative.")
    
    @classmethod
    def _validate_train_epochs(cls, value, options, instance):
        if not value > 0:
            raise ValueError("Value of `TRAIN_EPOCHS` must be greater than 0.")
    
    @classmethod
    def _validate_l1_depth(cls, value, options, instance):
        if not value > 0:
            raise ValueError("Value of `L1_DEPTH` must be greater than 0.")
        
    @classmethod
    def _validate_num_layers(cls, value, options, instance):
        if not value >= 2:
            raise ValueError("Value of `NUM_LAYERS` must be greater than or equal to 2.")
        
        if not instance.INPUT_IMAGE_SIZE/(2 ** (value - 1)) >= 4:
            raise ValueError("Value conflict between `INPUT_IMAGE_SIZE` and `NUM_LAYERS`! "
                            "INPUT_IMAGE_SIZE/2^(NUM_LAYERS - 1) must be greater than or equal to 4.")
    
    @classmethod
    def _validate_num_workers(cls, value, options, instance):
        if not value >= 0:
            raise ValueError("Value of `NUM_WORKERS` cannot be negative.")
    
    @classmethod
    def _validate_batch_size(cls, value, options, instance):
        if not value > 0:
            raise ValueError("Value of `BATCH_SIZE` must be greater than 0.")
    
    @classmethod
    def _validate_save_model(cls, value, options, instance):
        pass

    @classmethod
    def _validate_pt_weights(cls, value, options, instance):
        if instance.MODEL in ['AuxNet-C', 'AuxNet-D'] and value is None:
            raise ValueError("Value conflict between `MODEL` and `PT_WEIGHTS`! "
                            f"Model `{instance.MODEL}` requires `PT_WEIGHTS` to be specified.")

    # @classmethod
    # def _validate_load_layers(cls, value, options, instance):
    #     if not set(value).issubset(set(options)):
    #         raise ValueError(f"Value of `LOAD_LAYERS` must either contain any combination of {options} or be an empty list.")
    
    # @classmethod
    # def _validate_freeze_layers(cls, value, options, instance):
    #     if not set(value).issubset(set(options)):
    #         raise ValueError(f"Value of `FREEZE_LAYERS` must either contain any combination of {options} or be an empty list.")
    
    @classmethod
    def _validate_seg_classes(cls, value, options, instance):
        if not value > 0:
            raise ValueError("Value of `SEG_CLASSES` must be greater than 0.")
        
    @classmethod
    def _validate_seg_dropout(cls, value, options, instance):
        if not 0 <= value <= 1:
            raise ValueError("Value of `SEG_DROPOUT` must be between 0 and 1.")
    
    @classmethod
    def _validate_seg_threshold(cls, value, options, instance):
        if not 0 <= value <= 1:
            raise ValueError("Value of `SEG_THRESHOLD` must be between 0 and 1.")
    
    @classmethod
    def _validate_cls_classes(cls, value, options, instance):
        if not value > 0:
            raise ValueError("Value of `CLS_CLASSES` must be greater than 0.")
    
    @classmethod
    def _validate_cls_dropout(cls, value, options, instance):
        if not 0 <= value <= 1:
            raise ValueError("Value of `CLS_DROPOUT` must be between 0 and 1.")
    
    @classmethod
    def _validate_cls_threshold(cls, value, options, instance):
        if not 0 <= value <= 1:
            raise ValueError("Value of `CLS_THRESHOLD` must be between 0 and 1.")
    
    @classmethod
    def _validate_loss(cls, value, options, instance):
        if not value in options:
            raise ValueError(f"Value of `LOSS` must be one of {options}.")
    
    @classmethod
    def _validate_include_background(cls, value, options, instance):
        pass

    @classmethod
    def _validate_adaptive_weights(cls, value, options, instance):
        pass

    @classmethod
    def _validate_weights(cls, value, options, instance):
        pass

    @classmethod
    def _validate_clamp_delta(cls, value, options, instance):
        pass

    @classmethod
    def _validate_sigmoid_steepness(cls, value, options, instance):
        pass

    
    @classmethod
    def _validate_save_metric(cls, value, options, instance):
        if not value in options:
            raise ValueError(f"Value of `SAVE_METRIC` must be one of {options}.")
    
    @classmethod
    def _validate_boundary(cls, value, options, instance):
        pass
    
    @classmethod
    def _validate_test_split(cls, value, options, instance):
        if not 0 < value < 1:
            raise ValueError("Value of `TEST_SPLIT` must be between 0 and 1.")
    
    @classmethod
    def _validate_seed(cls, value, options, instance):
        pass
    
    @classmethod
    def _validate_cross_validation(cls, value, options, instance):
        pass
    
    @classmethod
    def _validate_fold_id(cls, value, options, instance):
        if not value >= 0:
            raise ValueError("Value of `FOLD_ID` cannot be negative.")
        
        if not value < int(1/instance.TEST_SPLIT):
            raise ValueError("Value conflict between `FOLD_ID` and `TEST_SPLIT`! "
                            "Value of `FOLD_ID` must be less than 1/TEST_SPLIT.")
    
    @classmethod
    def _validate_gpus(cls, value, options, instance):
        pass
    
    @classmethod
    def _validate_master_addr(cls, value, options, instance):
        pass

    @classmethod
    def _validate_master_port(cls, value, options, instance):
        pass
    
    @classmethod
    def _validate_log_wandb(cls, value, options, instance):
        pass

    @classmethod
    def _validate_log_local(cls, value, options, instance):
        pass