import os


class Hyperparams:
    def __init__(self, server, model_tag,
                 train_size, test_size,
                 ):
        self.inter_op_parallelism_threads = 5
        self.intra_op_parallelism_threads = 5
        self.per_process_gpu_memory_fraction = 1

        # # #######################nsh new user map prediction process (behavior embedding) #################
        self.process_type = 1

        self.output_unit = 812  # 所有地图id多分类, 类别0不参与训练
        self.output_sub_unit = 907
        self.batch_size = 8192  # 4096
        self.learning_rate = 0.003
        self.avg_time_gap = 217.54
        self.max_time_span = 7200.0
        self.fix_time_span = False
        self.maxlen = 10  # max length of Pad Sequence
        self.lastlen = 10
        self.dense_factor = 40  # the dense interpolation factor

        self.epochs = 30
        self.train_size = train_size
        self.test_size = test_size
        self.time_pred_loss_factor = 0.75

        # model parameter
        self.vocab_size = 812
        self.user_vocab_size = 1261
        self.min_cnt = 3  # words whose occurred less than min_cnt are encoded as <UNK>.
        self.hidden_units = 64
        self.num_blocks = 1  # number of encoder/decoder blocks
        self.num_decoder_blocks = 2
        self.num_encoder_blocks = 1
        self.num_heads = 8
        self.dropout_rate = 0.2
        self.portrait_vec_len = 256 + 7 * 24
        self.embedding_attention_size = 350
        self.daily_period_len = 6
        self.weekly_period_len = 4
        self.shift_hour = 6

        FILE_PATH = '/GameBERT/dataset/map-preload-prediction/train/{}'.format(server)
        TEST_FILE_PATH = '/GameBERT/dataset/map-preload-prediction/test/{}'.format(server)
        self.periodic_dict = os.path.join(FILE_PATH, 'all_role_id_period_dict.pkl')

        self.X_file_train = os.path.join(FILE_PATH, 'all_datatrain_seq11.pkl')
        self.Time_file_train = os.path.join(FILE_PATH, 'all_timetrain_seq11.pkl')
        self.Time_file_train_label = os.path.join(FILE_PATH, 'all_timetrain_label_seq11.pkl')
        self.y_file_train = os.path.join(FILE_PATH, 'all_labeltrain_seq11.pkl')

        self.X_file_test = os.path.join(TEST_FILE_PATH, 'all_datatrain_seq11.pkl')
        self.Time_file_test = os.path.join(TEST_FILE_PATH,  'all_timetrain_seq11.pkl')
        self.Time_file_test_label = os.path.join(TEST_FILE_PATH, 'all_timetrain_label_seq11.pkl')
        self.y_file_test = os.path.join(TEST_FILE_PATH, 'all_labeltrain_seq11.pkl')

        self.portrait_train = os.path.join(FILE_PATH, 'all_portrait_train_seq11_{}'.format(model_tag) + '.pkl')
        self.portrait_test = os.path.join(TEST_FILE_PATH, 'all_portrait_train_seq11_{}'.format(model_tag) + '.pkl')

        self.role_id_train = os.path.join(FILE_PATH, 'all_role_id_train_seq11.pkl')
        self.role_id_test = os.path.join(TEST_FILE_PATH, 'all_role_id_train_seq11.pkl')

        self.model_tag = model_tag



