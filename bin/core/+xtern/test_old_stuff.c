    printf("  Function \"code_image\".\n");

    printf("    With Linear nonlinearity, None polarity split and Subsample reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LINEAR;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NONE;
        enum reduce_type          reduce_type = SUBSAMPLE;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 18);
	assert(o_coeffs[0] == 1);
	assert(o_coeffs[1] == 3);
	assert(o_coeffs[2] == 3);
	assert(o_coeffs[3] == 1);
	assert(o_coeffs[4] == 2);
	assert(o_coeffs[5] == 3);
	assert(o_coeffs[6] == 3);
	assert(o_coeffs[7] == 1);
	assert(o_coeffs[8] == 3);
	assert(o_coeffs[9] == 3);
	assert(o_coeffs[10] == -1);
	assert(o_coeffs[11] == -1);
	assert(o_coeffs[12] == -3);
	assert(o_coeffs[13] == -3);
	assert(o_coeffs[14] == -2);
	assert(o_coeffs[15] == -1);
	assert(o_coeffs[16] == -3);
	assert(o_coeffs[17] == -2);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 16);
	assert(o_coeffs_idx[6] == 17);
	assert(o_coeffs_idx[7] == 18);
	assert(o_coeffs_idx[8] == 19);
	assert(o_coeffs_idx[9] == 20);
	assert(o_coeffs_idx[10] == 28);
	assert(o_coeffs_idx[11] == 29);
	assert(o_coeffs_idx[12] == 31);
	assert(o_coeffs_idx[13] == 32);
	assert(o_coeffs_idx[14] == 35);
	assert(o_coeffs_idx[15] == 45);
	assert(o_coeffs_idx[16] == 48);
	assert(o_coeffs_idx[17] == 51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -1);
	assert(((double*)coding_tmps)[9+2] == 2);
	assert(((double*)coding_tmps)[9+3] == -1);
	assert(((double*)coding_tmps)[9+4] == 1);
	assert(((double*)coding_tmps)[9+5] == -2);
	assert(((double*)coding_tmps)[9+6] == 2);
	assert(((double*)coding_tmps)[9+7] == -2);
	assert(((double*)coding_tmps)[9+8] == 3);
	assert(((double*)coding_tmps)[9+9] == -1);
	assert(((double*)coding_tmps)[9+10] == 3);
	assert(((double*)coding_tmps)[9+11] == -1);
	assert(((double*)coding_tmps)[9+12] == 3);
	assert(((double*)coding_tmps)[9+13] == 2);
	assert(((double*)coding_tmps)[9+14] == 3);
	assert(((double*)coding_tmps)[9+15] == -2);
	assert(((double*)coding_tmps)[9+16] == 3);
	assert(((double*)coding_tmps)[9+17] == -1);
	assert(((double*)coding_tmps)[9+18] == 2);
	assert(((double*)coding_tmps)[9+19] == -1);
	assert(((double*)coding_tmps)[9+20] == 3);
	assert(((double*)coding_tmps)[9+21] == -2);
	assert(((double*)coding_tmps)[9+22] == 2);
	assert(((double*)coding_tmps)[9+23] == -1);
	assert(((double*)coding_tmps)[9+24] == 1);
	assert(((double*)coding_tmps)[9+25] == -3);
	assert(((double*)coding_tmps)[9+26] == 2);
	assert(((double*)coding_tmps)[9+27] == -3);
	assert(((double*)coding_tmps)[9+28] == 1);
	assert(((double*)coding_tmps)[9+29] == -3);
	assert(((double*)coding_tmps)[9+30] == 2);
	assert(((double*)coding_tmps)[9+31] == -3);
	assert(((double*)coding_tmps)[9+32] == 3);
	assert(((double*)coding_tmps)[9+33] == -3);
	assert(((double*)coding_tmps)[9+34] == 3);
	assert(((double*)coding_tmps)[9+35] == -3);
	assert(((double*)coding_tmps)[9+36] == 3);
	assert(((double*)coding_tmps)[9+37] == -3);
	assert(((double*)coding_tmps)[9+38] == 3);
	assert(((double*)coding_tmps)[9+39] == -3);
	assert(((double*)coding_tmps)[9+40] == 3);
	assert(((double*)coding_tmps)[9+41] == -3);
	assert(((double*)coding_tmps)[9+42] == 2);
	assert(((double*)coding_tmps)[9+43] == -1);
	assert(((double*)coding_tmps)[9+44] == 3);
	assert(((double*)coding_tmps)[9+45] == -3);
	assert(((double*)coding_tmps)[9+46] == 2);
	assert(((double*)coding_tmps)[9+47] == -1);
	assert(((double*)coding_tmps)[9+48] == 1);
	assert(((double*)coding_tmps)[9+49] == -2);
	assert(((double*)coding_tmps)[9+50] == 2);
	assert(((double*)coding_tmps)[9+51] == -2);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -1);
	assert(((double*)coding_tmps)[9+54] == 2);
	assert(((double*)coding_tmps)[9+55] == -1);
	assert(((double*)coding_tmps)[9+56] == 2);
	assert(((double*)coding_tmps)[9+57] == 3);
	assert(((double*)coding_tmps)[9+58] == 3);
	assert(((double*)coding_tmps)[9+59] == -2);
	assert(((double*)coding_tmps)[9+60] == 3);
	assert(((double*)coding_tmps)[9+61] == -1);
	assert(((double*)coding_tmps)[9+62] == 3);
	assert(((double*)coding_tmps)[9+63] == -1);
	assert(((double*)coding_tmps)[9+64] == 3);
	assert(((double*)coding_tmps)[9+65] == -2);
	assert(((double*)coding_tmps)[9+66] == 2);
	assert(((double*)coding_tmps)[9+67] == -1);
       	assert(((double*)coding_tmps)[9+68] == 3);
	assert(((double*)coding_tmps)[9+69] == -1);
	assert(((double*)coding_tmps)[9+70] == 2);
	assert(((double*)coding_tmps)[9+71] == -1);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 3);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 3);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 0);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Linear nonlinearity, None polarity split and MaxNoSign reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LINEAR;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NONE;
        enum reduce_type          reduce_type = MAX_NO_SIGN;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 2);
	assert(o_coeffs[1] == 3);
	assert(o_coeffs[2] == 3);
	assert(o_coeffs[3] == 2);
	assert(o_coeffs[4] == 3);
	assert(o_coeffs[5] == 3);
	assert(o_coeffs[6] == 2);
	assert(o_coeffs[7] == 3);
	assert(o_coeffs[8] == 3);
	assert(o_coeffs[9] == 2);
	assert(o_coeffs[10] == 3);
	assert(o_coeffs[11] == 3);
	assert(o_coeffs[12] == 2);
	assert(o_coeffs[13] == 3);
	assert(o_coeffs[14] == 3);
	assert(o_coeffs[15] == 2);
	assert(o_coeffs[16] == 3);
	assert(o_coeffs[17] == 3);
	assert(o_coeffs[18] == 1);
	assert(o_coeffs[19] == 2);
	assert(o_coeffs[20] == 3);
	assert(o_coeffs[21] == 3);
	assert(o_coeffs[22] == 1);
	assert(o_coeffs[23] == 2);
	assert(o_coeffs[24] == 2);
	assert(o_coeffs[25] == 2);
	assert(o_coeffs[26] == 3);
	assert(o_coeffs[27] == 3);
	assert(o_coeffs[28] == 2);
	assert(o_coeffs[29] == 2);
	assert(o_coeffs[30] == 2);
	assert(o_coeffs[31] == 3);
	assert(o_coeffs[32] == 2);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 28);
	assert(o_coeffs_idx[19] == 29);
	assert(o_coeffs_idx[20] == 31);
	assert(o_coeffs_idx[21] == 32);
	assert(o_coeffs_idx[22] == 34);
	assert(o_coeffs_idx[23] == 35);
	assert(o_coeffs_idx[24] == 36);
	assert(o_coeffs_idx[25] == 37);
	assert(o_coeffs_idx[26] == 39);
	assert(o_coeffs_idx[27] == 40);
	assert(o_coeffs_idx[28] == 42);
	assert(o_coeffs_idx[29] == 43);
	assert(o_coeffs_idx[30] == 45);
	assert(o_coeffs_idx[31] == 48);
	assert(o_coeffs_idx[32] == 51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -1);
	assert(((double*)coding_tmps)[9+2] == 2);
	assert(((double*)coding_tmps)[9+3] == -1);
	assert(((double*)coding_tmps)[9+4] == 1);
	assert(((double*)coding_tmps)[9+5] == -2);
	assert(((double*)coding_tmps)[9+6] == 2);
	assert(((double*)coding_tmps)[9+7] == -2);
	assert(((double*)coding_tmps)[9+8] == 3);
	assert(((double*)coding_tmps)[9+9] == -1);
	assert(((double*)coding_tmps)[9+10] == 3);
	assert(((double*)coding_tmps)[9+11] == -1);
	assert(((double*)coding_tmps)[9+12] == 3);
	assert(((double*)coding_tmps)[9+13] == 2);
	assert(((double*)coding_tmps)[9+14] == 3);
	assert(((double*)coding_tmps)[9+15] == -2);
	assert(((double*)coding_tmps)[9+16] == 3);
	assert(((double*)coding_tmps)[9+17] == -1);
	assert(((double*)coding_tmps)[9+18] == 2);
	assert(((double*)coding_tmps)[9+19] == -1);
	assert(((double*)coding_tmps)[9+20] == 3);
	assert(((double*)coding_tmps)[9+21] == -2);
	assert(((double*)coding_tmps)[9+22] == 2);
	assert(((double*)coding_tmps)[9+23] == -1);
	assert(((double*)coding_tmps)[9+24] == 1);
	assert(((double*)coding_tmps)[9+25] == -3);
	assert(((double*)coding_tmps)[9+26] == 2);
	assert(((double*)coding_tmps)[9+27] == -3);
	assert(((double*)coding_tmps)[9+28] == 1);
	assert(((double*)coding_tmps)[9+29] == -3);
	assert(((double*)coding_tmps)[9+30] == 2);
	assert(((double*)coding_tmps)[9+31] == -3);
	assert(((double*)coding_tmps)[9+32] == 3);
	assert(((double*)coding_tmps)[9+33] == -3);
	assert(((double*)coding_tmps)[9+34] == 3);
	assert(((double*)coding_tmps)[9+35] == -3);
	assert(((double*)coding_tmps)[9+36] == 3);
	assert(((double*)coding_tmps)[9+37] == -3);
	assert(((double*)coding_tmps)[9+38] == 3);
	assert(((double*)coding_tmps)[9+39] == -3);
	assert(((double*)coding_tmps)[9+40] == 3);
	assert(((double*)coding_tmps)[9+41] == -3);
	assert(((double*)coding_tmps)[9+42] == 2);
	assert(((double*)coding_tmps)[9+43] == -1);
	assert(((double*)coding_tmps)[9+44] == 3);
	assert(((double*)coding_tmps)[9+45] == -3);
	assert(((double*)coding_tmps)[9+46] == 2);
	assert(((double*)coding_tmps)[9+47] == -1);
	assert(((double*)coding_tmps)[9+48] == 1);
	assert(((double*)coding_tmps)[9+49] == -2);
	assert(((double*)coding_tmps)[9+50] == 2);
	assert(((double*)coding_tmps)[9+51] == -2);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -1);
	assert(((double*)coding_tmps)[9+54] == 2);
	assert(((double*)coding_tmps)[9+55] == -1);
	assert(((double*)coding_tmps)[9+56] == 2);
	assert(((double*)coding_tmps)[9+57] == 3);
	assert(((double*)coding_tmps)[9+58] == 3);
	assert(((double*)coding_tmps)[9+59] == -2);
	assert(((double*)coding_tmps)[9+60] == 3);
	assert(((double*)coding_tmps)[9+61] == -1);
	assert(((double*)coding_tmps)[9+62] == 3);
	assert(((double*)coding_tmps)[9+63] == -1);
	assert(((double*)coding_tmps)[9+64] == 3);
	assert(((double*)coding_tmps)[9+65] == -2);
	assert(((double*)coding_tmps)[9+66] == 2);
	assert(((double*)coding_tmps)[9+67] == -1);
       	assert(((double*)coding_tmps)[9+68] == 3);
	assert(((double*)coding_tmps)[9+69] == -1);
	assert(((double*)coding_tmps)[9+70] == 2);
	assert(((double*)coding_tmps)[9+71] == -1);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 3);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 3);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Linear nonlinearity, None polarity split and MaxKeepSign reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LINEAR;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NONE;
        enum reduce_type          reduce_type = MAX_KEEP_SIGN;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 2);
	assert(o_coeffs[1] == 3);
	assert(o_coeffs[2] == 3);
	assert(o_coeffs[3] == 2);
	assert(o_coeffs[4] == 3);
	assert(o_coeffs[5] == 3);
	assert(o_coeffs[6] == 2);
	assert(o_coeffs[7] == 3);
	assert(o_coeffs[8] == 3);
	assert(o_coeffs[9] == 2);
	assert(o_coeffs[10] == 3);
	assert(o_coeffs[11] == 3);
	assert(o_coeffs[12] == 2);
	assert(o_coeffs[13] == 3);
	assert(o_coeffs[14] == 3);
	assert(o_coeffs[15] == 2);
	assert(o_coeffs[16] == 3);
	assert(o_coeffs[17] == 3);
	assert(o_coeffs[18] == -1);
	assert(o_coeffs[19] == -2);
	assert(o_coeffs[20] == -3);
	assert(o_coeffs[21] == -3);
	assert(o_coeffs[22] == -1);
	assert(o_coeffs[23] == -2);
	assert(o_coeffs[24] == -2);
	assert(o_coeffs[25] == -2);
	assert(o_coeffs[26] == -3);
	assert(o_coeffs[27] == -3);
	assert(o_coeffs[28] == -2);
	assert(o_coeffs[29] == -2);
	assert(o_coeffs[30] == -2);
	assert(o_coeffs[31] == -3);
	assert(o_coeffs[32] == -2);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 28);
	assert(o_coeffs_idx[19] == 29);
	assert(o_coeffs_idx[20] == 31);
	assert(o_coeffs_idx[21] == 32);
	assert(o_coeffs_idx[22] == 34);
	assert(o_coeffs_idx[23] == 35);
	assert(o_coeffs_idx[24] == 36);
	assert(o_coeffs_idx[25] == 37);
	assert(o_coeffs_idx[26] == 39);
	assert(o_coeffs_idx[27] == 40);
	assert(o_coeffs_idx[28] == 42);
	assert(o_coeffs_idx[29] == 43);
	assert(o_coeffs_idx[30] == 45);
	assert(o_coeffs_idx[31] == 48);
	assert(o_coeffs_idx[32] == 51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -1);
	assert(((double*)coding_tmps)[9+2] == 2);
	assert(((double*)coding_tmps)[9+3] == -1);
	assert(((double*)coding_tmps)[9+4] == 1);
	assert(((double*)coding_tmps)[9+5] == -2);
	assert(((double*)coding_tmps)[9+6] == 2);
	assert(((double*)coding_tmps)[9+7] == -2);
	assert(((double*)coding_tmps)[9+8] == 3);
	assert(((double*)coding_tmps)[9+9] == -1);
	assert(((double*)coding_tmps)[9+10] == 3);
	assert(((double*)coding_tmps)[9+11] == -1);
	assert(((double*)coding_tmps)[9+12] == 3);
	assert(((double*)coding_tmps)[9+13] == 2);
	assert(((double*)coding_tmps)[9+14] == 3);
	assert(((double*)coding_tmps)[9+15] == -2);
	assert(((double*)coding_tmps)[9+16] == 3);
	assert(((double*)coding_tmps)[9+17] == -1);
	assert(((double*)coding_tmps)[9+18] == 2);
	assert(((double*)coding_tmps)[9+19] == -1);
	assert(((double*)coding_tmps)[9+20] == 3);
	assert(((double*)coding_tmps)[9+21] == -2);
	assert(((double*)coding_tmps)[9+22] == 2);
	assert(((double*)coding_tmps)[9+23] == -1);
	assert(((double*)coding_tmps)[9+24] == 1);
	assert(((double*)coding_tmps)[9+25] == -3);
	assert(((double*)coding_tmps)[9+26] == 2);
	assert(((double*)coding_tmps)[9+27] == -3);
	assert(((double*)coding_tmps)[9+28] == 1);
	assert(((double*)coding_tmps)[9+29] == -3);
	assert(((double*)coding_tmps)[9+30] == 2);
	assert(((double*)coding_tmps)[9+31] == -3);
	assert(((double*)coding_tmps)[9+32] == 3);
	assert(((double*)coding_tmps)[9+33] == -3);
	assert(((double*)coding_tmps)[9+34] == 3);
	assert(((double*)coding_tmps)[9+35] == -3);
	assert(((double*)coding_tmps)[9+36] == 3);
	assert(((double*)coding_tmps)[9+37] == -3);
	assert(((double*)coding_tmps)[9+38] == 3);
	assert(((double*)coding_tmps)[9+39] == -3);
	assert(((double*)coding_tmps)[9+40] == 3);
	assert(((double*)coding_tmps)[9+41] == -3);
	assert(((double*)coding_tmps)[9+42] == 2);
	assert(((double*)coding_tmps)[9+43] == -1);
	assert(((double*)coding_tmps)[9+44] == 3);
	assert(((double*)coding_tmps)[9+45] == -3);
	assert(((double*)coding_tmps)[9+46] == 2);
	assert(((double*)coding_tmps)[9+47] == -1);
	assert(((double*)coding_tmps)[9+48] == 1);
	assert(((double*)coding_tmps)[9+49] == -2);
	assert(((double*)coding_tmps)[9+50] == 2);
	assert(((double*)coding_tmps)[9+51] == -2);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -1);
	assert(((double*)coding_tmps)[9+54] == 2);
	assert(((double*)coding_tmps)[9+55] == -1);
	assert(((double*)coding_tmps)[9+56] == 2);
	assert(((double*)coding_tmps)[9+57] == 3);
	assert(((double*)coding_tmps)[9+58] == 3);
	assert(((double*)coding_tmps)[9+59] == -2);
	assert(((double*)coding_tmps)[9+60] == 3);
	assert(((double*)coding_tmps)[9+61] == -1);
	assert(((double*)coding_tmps)[9+62] == 3);
	assert(((double*)coding_tmps)[9+63] == -1);
	assert(((double*)coding_tmps)[9+64] == 3);
	assert(((double*)coding_tmps)[9+65] == -2);
	assert(((double*)coding_tmps)[9+66] == 2);
	assert(((double*)coding_tmps)[9+67] == -1);
       	assert(((double*)coding_tmps)[9+68] == 3);
	assert(((double*)coding_tmps)[9+69] == -1);
	assert(((double*)coding_tmps)[9+70] == 2);
	assert(((double*)coding_tmps)[9+71] == -1);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 3);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 3);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Linear nonlinearity, None polarity split and SumAbs reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LINEAR;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NONE;
        enum reduce_type          reduce_type = SUM_ABS;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 1 + 1 + 2);
	assert(o_coeffs[1] == 3 + 3);
	assert(o_coeffs[2] == 2 + 3);
	assert(o_coeffs[3] == 1 + 1 + 2);
	assert(o_coeffs[4] == 2 + 3 + 3);
	assert(o_coeffs[5] == 2 + 3);
	assert(o_coeffs[6] == 1 + 2);
	assert(o_coeffs[7] == 3 + 3);
	assert(o_coeffs[8] == 2 + 3);
	assert(o_coeffs[9] == 2);
	assert(o_coeffs[10] == 3 + 3);
	assert(o_coeffs[11] == 2 + 3);
	assert(o_coeffs[12] == 1 + 2);
	assert(o_coeffs[13] == 2 + 3 + 3);
	assert(o_coeffs[14] == 2 + 3);
	assert(o_coeffs[15] == 2);
	assert(o_coeffs[16] == 3 + 3);
	assert(o_coeffs[17] == 2 + 3);
	assert(o_coeffs[18] == 1 + 1);
	assert(o_coeffs[19] == 1 + 1 + 1 + 2);
	assert(o_coeffs[20] == 3 + 3);
	assert(o_coeffs[21] == 1 + 1 + 3 + 3);
	assert(o_coeffs[22] == 1 + 1);
	assert(o_coeffs[23] == 1 + 1 + 1 + 2);
	assert(o_coeffs[24] == 1 + 2);
	assert(o_coeffs[25] == 2);
	assert(o_coeffs[26] == 3 + 3);
	assert(o_coeffs[27] == 3 + 3);
	assert(o_coeffs[28] == 1 + 2);
	assert(o_coeffs[29] == 2);
	assert(o_coeffs[30] == 1 + 2);
	assert(o_coeffs[31] == 3 + 3);
	assert(o_coeffs[32] == 1 + 2);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 28);
	assert(o_coeffs_idx[19] == 29);
	assert(o_coeffs_idx[20] == 31);
	assert(o_coeffs_idx[21] == 32);
	assert(o_coeffs_idx[22] == 34);
	assert(o_coeffs_idx[23] == 35);
	assert(o_coeffs_idx[24] == 36);
	assert(o_coeffs_idx[25] == 37);
	assert(o_coeffs_idx[26] == 39);
	assert(o_coeffs_idx[27] == 40);
	assert(o_coeffs_idx[28] == 42);
	assert(o_coeffs_idx[29] == 43);
	assert(o_coeffs_idx[30] == 45);
	assert(o_coeffs_idx[31] == 48);
	assert(o_coeffs_idx[32] == 51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -1);
	assert(((double*)coding_tmps)[9+2] == 2);
	assert(((double*)coding_tmps)[9+3] == -1);
	assert(((double*)coding_tmps)[9+4] == 1);
	assert(((double*)coding_tmps)[9+5] == -2);
	assert(((double*)coding_tmps)[9+6] == 2);
	assert(((double*)coding_tmps)[9+7] == -2);
	assert(((double*)coding_tmps)[9+8] == 3);
	assert(((double*)coding_tmps)[9+9] == -1);
	assert(((double*)coding_tmps)[9+10] == 3);
	assert(((double*)coding_tmps)[9+11] == -1);
	assert(((double*)coding_tmps)[9+12] == 3);
	assert(((double*)coding_tmps)[9+13] == 2);
	assert(((double*)coding_tmps)[9+14] == 3);
	assert(((double*)coding_tmps)[9+15] == -2);
	assert(((double*)coding_tmps)[9+16] == 3);
	assert(((double*)coding_tmps)[9+17] == -1);
	assert(((double*)coding_tmps)[9+18] == 2);
	assert(((double*)coding_tmps)[9+19] == -1);
	assert(((double*)coding_tmps)[9+20] == 3);
	assert(((double*)coding_tmps)[9+21] == -2);
	assert(((double*)coding_tmps)[9+22] == 2);
	assert(((double*)coding_tmps)[9+23] == -1);
	assert(((double*)coding_tmps)[9+24] == 1);
	assert(((double*)coding_tmps)[9+25] == -3);
	assert(((double*)coding_tmps)[9+26] == 2);
	assert(((double*)coding_tmps)[9+27] == -3);
	assert(((double*)coding_tmps)[9+28] == 1);
	assert(((double*)coding_tmps)[9+29] == -3);
	assert(((double*)coding_tmps)[9+30] == 2);
	assert(((double*)coding_tmps)[9+31] == -3);
	assert(((double*)coding_tmps)[9+32] == 3);
	assert(((double*)coding_tmps)[9+33] == -3);
	assert(((double*)coding_tmps)[9+34] == 3);
	assert(((double*)coding_tmps)[9+35] == -3);
	assert(((double*)coding_tmps)[9+36] == 3);
	assert(((double*)coding_tmps)[9+37] == -3);
	assert(((double*)coding_tmps)[9+38] == 3);
	assert(((double*)coding_tmps)[9+39] == -3);
	assert(((double*)coding_tmps)[9+40] == 3);
	assert(((double*)coding_tmps)[9+41] == -3);
	assert(((double*)coding_tmps)[9+42] == 2);
	assert(((double*)coding_tmps)[9+43] == -1);
	assert(((double*)coding_tmps)[9+44] == 3);
	assert(((double*)coding_tmps)[9+45] == -3);
	assert(((double*)coding_tmps)[9+46] == 2);
	assert(((double*)coding_tmps)[9+47] == -1);
	assert(((double*)coding_tmps)[9+48] == 1);
	assert(((double*)coding_tmps)[9+49] == -2);
	assert(((double*)coding_tmps)[9+50] == 2);
	assert(((double*)coding_tmps)[9+51] == -2);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -1);
	assert(((double*)coding_tmps)[9+54] == 2);
	assert(((double*)coding_tmps)[9+55] == -1);
	assert(((double*)coding_tmps)[9+56] == 2);
	assert(((double*)coding_tmps)[9+57] == 3);
	assert(((double*)coding_tmps)[9+58] == 3);
	assert(((double*)coding_tmps)[9+59] == -2);
	assert(((double*)coding_tmps)[9+60] == 3);
	assert(((double*)coding_tmps)[9+61] == -1);
	assert(((double*)coding_tmps)[9+62] == 3);
	assert(((double*)coding_tmps)[9+63] == -1);
	assert(((double*)coding_tmps)[9+64] == 3);
	assert(((double*)coding_tmps)[9+65] == -2);
	assert(((double*)coding_tmps)[9+66] == 2);
	assert(((double*)coding_tmps)[9+67] == -1);
       	assert(((double*)coding_tmps)[9+68] == 3);
	assert(((double*)coding_tmps)[9+69] == -1);
	assert(((double*)coding_tmps)[9+70] == 2);
	assert(((double*)coding_tmps)[9+71] == -1);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 3);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 3);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Linear nonlinearity, None polarity split and SumSqr reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LINEAR;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NONE;
        enum reduce_type          reduce_type = SUM_SQR;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 1*1 + 1*1 + 2*2);
	assert(o_coeffs[1] == 3*3 + 3*3);
	assert(o_coeffs[2] == 3*3 + 2*2);
	assert(o_coeffs[3] == 1*1 + 1*1 + 2*2);
	assert(o_coeffs[4] == 2*2 + 3*3 + 3*3);
	assert(o_coeffs[5] == 2*2 + 3*3);
	assert(o_coeffs[6] == 1*1 + 2*2);
	assert(o_coeffs[7] == 3*3 + 3*3);
	assert(o_coeffs[8] == 2*2 + 3*3);
	assert(o_coeffs[9] == 2*2);
	assert(o_coeffs[10] == 3*3 + 3*3);
	assert(o_coeffs[11] == 2*2 + 3*3);
	assert(o_coeffs[12] == 1*1 + 2*2);
	assert(o_coeffs[13] == 2*2 + 3*3 + 3*3);
	assert(o_coeffs[14] == 2*2 + 3*3);
	assert(o_coeffs[15] == 2*2);
	assert(o_coeffs[16] == 3*3 + 3*3);
	assert(o_coeffs[17] == 2*2 + 3*3);
	assert(o_coeffs[18] == 1*1 + 1*1);
	assert(o_coeffs[19] == 1*1 + 1*1 + 1*1 + 2*2);
	assert(o_coeffs[20] == 3*3 + 3*3);
	assert(o_coeffs[21] == 1*1 + 1*1 + 3*3 + 3*3);
	assert(o_coeffs[22] == 1*1 + 1*1);
	assert(o_coeffs[23] == 1*1 + 1*1 + 1*1 + 2*2);
	assert(o_coeffs[24] == 1*1 + 2*2);
	assert(o_coeffs[25] == 2*2);
	assert(o_coeffs[26] == 3*3 + 3*3);
	assert(o_coeffs[27] == 3*3 + 3*3);
	assert(o_coeffs[28] == 1*1 + 2*2);
	assert(o_coeffs[29] == 2*2);
	assert(o_coeffs[30] == 1*1 + 2*2);
	assert(o_coeffs[31] == 3*3 + 3*3);
	assert(o_coeffs[32] == 1*1 + 2*2);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 28);
	assert(o_coeffs_idx[19] == 29);
	assert(o_coeffs_idx[20] == 31);
	assert(o_coeffs_idx[21] == 32);
	assert(o_coeffs_idx[22] == 34);
	assert(o_coeffs_idx[23] == 35);
	assert(o_coeffs_idx[24] == 36);
	assert(o_coeffs_idx[25] == 37);
	assert(o_coeffs_idx[26] == 39);
	assert(o_coeffs_idx[27] == 40);
	assert(o_coeffs_idx[28] == 42);
	assert(o_coeffs_idx[29] == 43);
	assert(o_coeffs_idx[30] == 45);
	assert(o_coeffs_idx[31] == 48);
	assert(o_coeffs_idx[32] == 51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -1);
	assert(((double*)coding_tmps)[9+2] == 2);
	assert(((double*)coding_tmps)[9+3] == -1);
	assert(((double*)coding_tmps)[9+4] == 1);
	assert(((double*)coding_tmps)[9+5] == -2);
	assert(((double*)coding_tmps)[9+6] == 2);
	assert(((double*)coding_tmps)[9+7] == -2);
	assert(((double*)coding_tmps)[9+8] == 3);
	assert(((double*)coding_tmps)[9+9] == -1);
	assert(((double*)coding_tmps)[9+10] == 3);
	assert(((double*)coding_tmps)[9+11] == -1);
	assert(((double*)coding_tmps)[9+12] == 3);
	assert(((double*)coding_tmps)[9+13] == 2);
	assert(((double*)coding_tmps)[9+14] == 3);
	assert(((double*)coding_tmps)[9+15] == -2);
	assert(((double*)coding_tmps)[9+16] == 3);
	assert(((double*)coding_tmps)[9+17] == -1);
	assert(((double*)coding_tmps)[9+18] == 2);
	assert(((double*)coding_tmps)[9+19] == -1);
	assert(((double*)coding_tmps)[9+20] == 3);
	assert(((double*)coding_tmps)[9+21] == -2);
	assert(((double*)coding_tmps)[9+22] == 2);
	assert(((double*)coding_tmps)[9+23] == -1);
	assert(((double*)coding_tmps)[9+24] == 1);
	assert(((double*)coding_tmps)[9+25] == -3);
	assert(((double*)coding_tmps)[9+26] == 2);
	assert(((double*)coding_tmps)[9+27] == -3);
	assert(((double*)coding_tmps)[9+28] == 1);
	assert(((double*)coding_tmps)[9+29] == -3);
	assert(((double*)coding_tmps)[9+30] == 2);
	assert(((double*)coding_tmps)[9+31] == -3);
	assert(((double*)coding_tmps)[9+32] == 3);
	assert(((double*)coding_tmps)[9+33] == -3);
	assert(((double*)coding_tmps)[9+34] == 3);
	assert(((double*)coding_tmps)[9+35] == -3);
	assert(((double*)coding_tmps)[9+36] == 3);
	assert(((double*)coding_tmps)[9+37] == -3);
	assert(((double*)coding_tmps)[9+38] == 3);
	assert(((double*)coding_tmps)[9+39] == -3);
	assert(((double*)coding_tmps)[9+40] == 3);
	assert(((double*)coding_tmps)[9+41] == -3);
	assert(((double*)coding_tmps)[9+42] == 2);
	assert(((double*)coding_tmps)[9+43] == -1);
	assert(((double*)coding_tmps)[9+44] == 3);
	assert(((double*)coding_tmps)[9+45] == -3);
	assert(((double*)coding_tmps)[9+46] == 2);
	assert(((double*)coding_tmps)[9+47] == -1);
	assert(((double*)coding_tmps)[9+48] == 1);
	assert(((double*)coding_tmps)[9+49] == -2);
	assert(((double*)coding_tmps)[9+50] == 2);
	assert(((double*)coding_tmps)[9+51] == -2);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -1);
	assert(((double*)coding_tmps)[9+54] == 2);
	assert(((double*)coding_tmps)[9+55] == -1);
	assert(((double*)coding_tmps)[9+56] == 2);
	assert(((double*)coding_tmps)[9+57] == 3);
	assert(((double*)coding_tmps)[9+58] == 3);
	assert(((double*)coding_tmps)[9+59] == -2);
	assert(((double*)coding_tmps)[9+60] == 3);
	assert(((double*)coding_tmps)[9+61] == -1);
	assert(((double*)coding_tmps)[9+62] == 3);
	assert(((double*)coding_tmps)[9+63] == -1);
	assert(((double*)coding_tmps)[9+64] == 3);
	assert(((double*)coding_tmps)[9+65] == -2);
	assert(((double*)coding_tmps)[9+66] == 2);
	assert(((double*)coding_tmps)[9+67] == -1);
       	assert(((double*)coding_tmps)[9+68] == 3);
	assert(((double*)coding_tmps)[9+69] == -1);
	assert(((double*)coding_tmps)[9+70] == 2);
	assert(((double*)coding_tmps)[9+71] == -1);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 3);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 3);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Linear nonlinearity, NoSign polarity split and Subsample reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LINEAR;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NO_SIGN;
        enum reduce_type          reduce_type = SUBSAMPLE;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 18);
	assert(o_coeffs[0] == 1);
	assert(o_coeffs[1] == 3);
	assert(o_coeffs[2] == 3);
	assert(o_coeffs[3] == 1);
	assert(o_coeffs[4] == 2);
	assert(o_coeffs[5] == 3);
	assert(o_coeffs[6] == 3);
	assert(o_coeffs[7] == 1);
	assert(o_coeffs[8] == 3);
	assert(o_coeffs[9] == 3);
	assert(o_coeffs[10] == 1);
	assert(o_coeffs[11] == 1);
	assert(o_coeffs[12] == 3);
	assert(o_coeffs[13] == 3);
	assert(o_coeffs[14] == 2);
	assert(o_coeffs[15] == 1);
	assert(o_coeffs[16] == 3);
	assert(o_coeffs[17] == 2);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 16);
	assert(o_coeffs_idx[6] == 17);
	assert(o_coeffs_idx[7] == 18);
	assert(o_coeffs_idx[8] == 19);
	assert(o_coeffs_idx[9] == 20);
	assert(o_coeffs_idx[10] == 54+28);
	assert(o_coeffs_idx[11] == 54+29);
	assert(o_coeffs_idx[12] == 54+31);
	assert(o_coeffs_idx[13] == 54+32);
	assert(o_coeffs_idx[14] == 54+35);
	assert(o_coeffs_idx[15] == 54+45);
	assert(o_coeffs_idx[16] == 54+48);
	assert(o_coeffs_idx[17] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == 1);
	assert(((double*)coding_tmps)[9+2] == 2);
	assert(((double*)coding_tmps)[9+3] == 1);
	assert(((double*)coding_tmps)[9+4] == 1);
	assert(((double*)coding_tmps)[9+5] == 2);
	assert(((double*)coding_tmps)[9+6] == 2);
	assert(((double*)coding_tmps)[9+7] == 2);
	assert(((double*)coding_tmps)[9+8] == 3);
	assert(((double*)coding_tmps)[9+9] == 1);
	assert(((double*)coding_tmps)[9+10] == 3);
	assert(((double*)coding_tmps)[9+11] == 1);
	assert(((double*)coding_tmps)[9+12] == 3);
	assert(((double*)coding_tmps)[9+13] == 2);
	assert(((double*)coding_tmps)[9+14] == 3);
	assert(((double*)coding_tmps)[9+15] == 2);
	assert(((double*)coding_tmps)[9+16] == 3);
	assert(((double*)coding_tmps)[9+17] == 1);
	assert(((double*)coding_tmps)[9+18] == 2);
	assert(((double*)coding_tmps)[9+19] == 1);
	assert(((double*)coding_tmps)[9+20] == 3);
	assert(((double*)coding_tmps)[9+21] == 2);
	assert(((double*)coding_tmps)[9+22] == 2);
	assert(((double*)coding_tmps)[9+23] == 1);
	assert(((double*)coding_tmps)[9+24] == 1);
	assert(((double*)coding_tmps)[9+25] == 3);
	assert(((double*)coding_tmps)[9+26] == 2);
	assert(((double*)coding_tmps)[9+27] == 3);
	assert(((double*)coding_tmps)[9+28] == 1);
	assert(((double*)coding_tmps)[9+29] == 3);
	assert(((double*)coding_tmps)[9+30] == 2);
	assert(((double*)coding_tmps)[9+31] == 3);
	assert(((double*)coding_tmps)[9+32] == 3);
	assert(((double*)coding_tmps)[9+33] == 3);
	assert(((double*)coding_tmps)[9+34] == 3);
	assert(((double*)coding_tmps)[9+35] == 3);
	assert(((double*)coding_tmps)[9+36] == 3);
	assert(((double*)coding_tmps)[9+37] == 3);
	assert(((double*)coding_tmps)[9+38] == 3);
	assert(((double*)coding_tmps)[9+39] == 3);
	assert(((double*)coding_tmps)[9+40] == 3);
	assert(((double*)coding_tmps)[9+41] == 3);
	assert(((double*)coding_tmps)[9+42] == 2);
	assert(((double*)coding_tmps)[9+43] == 1);
	assert(((double*)coding_tmps)[9+44] == 3);
	assert(((double*)coding_tmps)[9+45] == 3);
	assert(((double*)coding_tmps)[9+46] == 2);
	assert(((double*)coding_tmps)[9+47] == 1);
	assert(((double*)coding_tmps)[9+48] == 1);
	assert(((double*)coding_tmps)[9+49] == 2);
	assert(((double*)coding_tmps)[9+50] == 2);
	assert(((double*)coding_tmps)[9+51] == 2);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == 1);
	assert(((double*)coding_tmps)[9+54] == 2);
	assert(((double*)coding_tmps)[9+55] == 1);
	assert(((double*)coding_tmps)[9+56] == 2);
	assert(((double*)coding_tmps)[9+57] == 3);
	assert(((double*)coding_tmps)[9+58] == 3);
	assert(((double*)coding_tmps)[9+59] == 2);
	assert(((double*)coding_tmps)[9+60] == 3);
	assert(((double*)coding_tmps)[9+61] == 1);
	assert(((double*)coding_tmps)[9+62] == 3);
	assert(((double*)coding_tmps)[9+63] == 1);
	assert(((double*)coding_tmps)[9+64] == 3);
	assert(((double*)coding_tmps)[9+65] == 2);
	assert(((double*)coding_tmps)[9+66] == 2);
	assert(((double*)coding_tmps)[9+67] == 1);
       	assert(((double*)coding_tmps)[9+68] == 3);
	assert(((double*)coding_tmps)[9+69] == 1);
	assert(((double*)coding_tmps)[9+70] == 2);
	assert(((double*)coding_tmps)[9+71] == 1);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 0);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Linear nonlinearity, NoSign polarity split and MaxNoSign reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LINEAR;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NO_SIGN;
        enum reduce_type          reduce_type = MAX_NO_SIGN;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 2);
	assert(o_coeffs[1] == 3);
	assert(o_coeffs[2] == 3);
	assert(o_coeffs[3] == 2);
	assert(o_coeffs[4] == 3);
	assert(o_coeffs[5] == 3);
	assert(o_coeffs[6] == 2);
	assert(o_coeffs[7] == 3);
	assert(o_coeffs[8] == 3);
	assert(o_coeffs[9] == 2);
	assert(o_coeffs[10] == 3);
	assert(o_coeffs[11] == 3);
	assert(o_coeffs[12] == 2);
	assert(o_coeffs[13] == 3);
	assert(o_coeffs[14] == 3);
	assert(o_coeffs[15] == 2);
	assert(o_coeffs[16] == 3);
	assert(o_coeffs[17] == 3);
	assert(o_coeffs[18] == 1);
	assert(o_coeffs[19] == 2);
	assert(o_coeffs[20] == 3);
	assert(o_coeffs[21] == 3);
	assert(o_coeffs[22] == 1);
	assert(o_coeffs[23] == 2);
	assert(o_coeffs[24] == 2);
	assert(o_coeffs[25] == 2);
	assert(o_coeffs[26] == 3);
	assert(o_coeffs[27] == 3);
	assert(o_coeffs[28] == 2);
	assert(o_coeffs[29] == 2);
	assert(o_coeffs[30] == 2);
	assert(o_coeffs[31] == 3);
	assert(o_coeffs[32] == 2);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == 1);
	assert(((double*)coding_tmps)[9+2] == 2);
	assert(((double*)coding_tmps)[9+3] == 1);
	assert(((double*)coding_tmps)[9+4] == 1);
	assert(((double*)coding_tmps)[9+5] == 2);
	assert(((double*)coding_tmps)[9+6] == 2);
	assert(((double*)coding_tmps)[9+7] == 2);
	assert(((double*)coding_tmps)[9+8] == 3);
	assert(((double*)coding_tmps)[9+9] == 1);
	assert(((double*)coding_tmps)[9+10] == 3);
	assert(((double*)coding_tmps)[9+11] == 1);
	assert(((double*)coding_tmps)[9+12] == 3);
	assert(((double*)coding_tmps)[9+13] == 2);
	assert(((double*)coding_tmps)[9+14] == 3);
	assert(((double*)coding_tmps)[9+15] == 2);
	assert(((double*)coding_tmps)[9+16] == 3);
	assert(((double*)coding_tmps)[9+17] == 1);
	assert(((double*)coding_tmps)[9+18] == 2);
	assert(((double*)coding_tmps)[9+19] == 1);
	assert(((double*)coding_tmps)[9+20] == 3);
	assert(((double*)coding_tmps)[9+21] == 2);
	assert(((double*)coding_tmps)[9+22] == 2);
	assert(((double*)coding_tmps)[9+23] == 1);
	assert(((double*)coding_tmps)[9+24] == 1);
	assert(((double*)coding_tmps)[9+25] == 3);
	assert(((double*)coding_tmps)[9+26] == 2);
	assert(((double*)coding_tmps)[9+27] == 3);
	assert(((double*)coding_tmps)[9+28] == 1);
	assert(((double*)coding_tmps)[9+29] == 3);
	assert(((double*)coding_tmps)[9+30] == 2);
	assert(((double*)coding_tmps)[9+31] == 3);
	assert(((double*)coding_tmps)[9+32] == 3);
	assert(((double*)coding_tmps)[9+33] == 3);
	assert(((double*)coding_tmps)[9+34] == 3);
	assert(((double*)coding_tmps)[9+35] == 3);
	assert(((double*)coding_tmps)[9+36] == 3);
	assert(((double*)coding_tmps)[9+37] == 3);
	assert(((double*)coding_tmps)[9+38] == 3);
	assert(((double*)coding_tmps)[9+39] == 3);
	assert(((double*)coding_tmps)[9+40] == 3);
	assert(((double*)coding_tmps)[9+41] == 3);
	assert(((double*)coding_tmps)[9+42] == 2);
	assert(((double*)coding_tmps)[9+43] == 1);
	assert(((double*)coding_tmps)[9+44] == 3);
	assert(((double*)coding_tmps)[9+45] == 3);
	assert(((double*)coding_tmps)[9+46] == 2);
	assert(((double*)coding_tmps)[9+47] == 1);
	assert(((double*)coding_tmps)[9+48] == 1);
	assert(((double*)coding_tmps)[9+49] == 2);
	assert(((double*)coding_tmps)[9+50] == 2);
	assert(((double*)coding_tmps)[9+51] == 2);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == 1);
	assert(((double*)coding_tmps)[9+54] == 2);
	assert(((double*)coding_tmps)[9+55] == 1);
	assert(((double*)coding_tmps)[9+56] == 2);
	assert(((double*)coding_tmps)[9+57] == 3);
	assert(((double*)coding_tmps)[9+58] == 3);
	assert(((double*)coding_tmps)[9+59] == 2);
	assert(((double*)coding_tmps)[9+60] == 3);
	assert(((double*)coding_tmps)[9+61] == 1);
	assert(((double*)coding_tmps)[9+62] == 3);
	assert(((double*)coding_tmps)[9+63] == 1);
	assert(((double*)coding_tmps)[9+64] == 3);
	assert(((double*)coding_tmps)[9+65] == 2);
	assert(((double*)coding_tmps)[9+66] == 2);
	assert(((double*)coding_tmps)[9+67] == 1);
       	assert(((double*)coding_tmps)[9+68] == 3);
	assert(((double*)coding_tmps)[9+69] == 1);
	assert(((double*)coding_tmps)[9+70] == 2);
	assert(((double*)coding_tmps)[9+71] == 1);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Linear nonlinearity, NoSign polarity split and MaxKeepSign reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LINEAR;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NO_SIGN;
        enum reduce_type          reduce_type = MAX_KEEP_SIGN;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 2);
	assert(o_coeffs[1] == 3);
	assert(o_coeffs[2] == 3);
	assert(o_coeffs[3] == 2);
	assert(o_coeffs[4] == 3);
	assert(o_coeffs[5] == 3);
	assert(o_coeffs[6] == 2);
	assert(o_coeffs[7] == 3);
	assert(o_coeffs[8] == 3);
	assert(o_coeffs[9] == 2);
	assert(o_coeffs[10] == 3);
	assert(o_coeffs[11] == 3);
	assert(o_coeffs[12] == 2);
	assert(o_coeffs[13] == 3);
	assert(o_coeffs[14] == 3);
	assert(o_coeffs[15] == 2);
	assert(o_coeffs[16] == 3);
	assert(o_coeffs[17] == 3);
	assert(o_coeffs[18] == 1);
	assert(o_coeffs[19] == 2);
	assert(o_coeffs[20] == 3);
	assert(o_coeffs[21] == 3);
	assert(o_coeffs[22] == 1);
	assert(o_coeffs[23] == 2);
	assert(o_coeffs[24] == 2);
	assert(o_coeffs[25] == 2);
	assert(o_coeffs[26] == 3);
	assert(o_coeffs[27] == 3);
	assert(o_coeffs[28] == 2);
	assert(o_coeffs[29] == 2);
	assert(o_coeffs[30] == 2);
	assert(o_coeffs[31] == 3);
	assert(o_coeffs[32] == 2);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == 1);
	assert(((double*)coding_tmps)[9+2] == 2);
	assert(((double*)coding_tmps)[9+3] == 1);
	assert(((double*)coding_tmps)[9+4] == 1);
	assert(((double*)coding_tmps)[9+5] == 2);
	assert(((double*)coding_tmps)[9+6] == 2);
	assert(((double*)coding_tmps)[9+7] == 2);
	assert(((double*)coding_tmps)[9+8] == 3);
	assert(((double*)coding_tmps)[9+9] == 1);
	assert(((double*)coding_tmps)[9+10] == 3);
	assert(((double*)coding_tmps)[9+11] == 1);
	assert(((double*)coding_tmps)[9+12] == 3);
	assert(((double*)coding_tmps)[9+13] == 2);
	assert(((double*)coding_tmps)[9+14] == 3);
	assert(((double*)coding_tmps)[9+15] == 2);
	assert(((double*)coding_tmps)[9+16] == 3);
	assert(((double*)coding_tmps)[9+17] == 1);
	assert(((double*)coding_tmps)[9+18] == 2);
	assert(((double*)coding_tmps)[9+19] == 1);
	assert(((double*)coding_tmps)[9+20] == 3);
	assert(((double*)coding_tmps)[9+21] == 2);
	assert(((double*)coding_tmps)[9+22] == 2);
	assert(((double*)coding_tmps)[9+23] == 1);
	assert(((double*)coding_tmps)[9+24] == 1);
	assert(((double*)coding_tmps)[9+25] == 3);
	assert(((double*)coding_tmps)[9+26] == 2);
	assert(((double*)coding_tmps)[9+27] == 3);
	assert(((double*)coding_tmps)[9+28] == 1);
	assert(((double*)coding_tmps)[9+29] == 3);
	assert(((double*)coding_tmps)[9+30] == 2);
	assert(((double*)coding_tmps)[9+31] == 3);
	assert(((double*)coding_tmps)[9+32] == 3);
	assert(((double*)coding_tmps)[9+33] == 3);
	assert(((double*)coding_tmps)[9+34] == 3);
	assert(((double*)coding_tmps)[9+35] == 3);
	assert(((double*)coding_tmps)[9+36] == 3);
	assert(((double*)coding_tmps)[9+37] == 3);
	assert(((double*)coding_tmps)[9+38] == 3);
	assert(((double*)coding_tmps)[9+39] == 3);
	assert(((double*)coding_tmps)[9+40] == 3);
	assert(((double*)coding_tmps)[9+41] == 3);
	assert(((double*)coding_tmps)[9+42] == 2);
	assert(((double*)coding_tmps)[9+43] == 1);
	assert(((double*)coding_tmps)[9+44] == 3);
	assert(((double*)coding_tmps)[9+45] == 3);
	assert(((double*)coding_tmps)[9+46] == 2);
	assert(((double*)coding_tmps)[9+47] == 1);
	assert(((double*)coding_tmps)[9+48] == 1);
	assert(((double*)coding_tmps)[9+49] == 2);
	assert(((double*)coding_tmps)[9+50] == 2);
	assert(((double*)coding_tmps)[9+51] == 2);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == 1);
	assert(((double*)coding_tmps)[9+54] == 2);
	assert(((double*)coding_tmps)[9+55] == 1);
	assert(((double*)coding_tmps)[9+56] == 2);
	assert(((double*)coding_tmps)[9+57] == 3);
	assert(((double*)coding_tmps)[9+58] == 3);
	assert(((double*)coding_tmps)[9+59] == 2);
	assert(((double*)coding_tmps)[9+60] == 3);
	assert(((double*)coding_tmps)[9+61] == 1);
	assert(((double*)coding_tmps)[9+62] == 3);
	assert(((double*)coding_tmps)[9+63] == 1);
	assert(((double*)coding_tmps)[9+64] == 3);
	assert(((double*)coding_tmps)[9+65] == 2);
	assert(((double*)coding_tmps)[9+66] == 2);
	assert(((double*)coding_tmps)[9+67] == 1);
       	assert(((double*)coding_tmps)[9+68] == 3);
	assert(((double*)coding_tmps)[9+69] == 1);
	assert(((double*)coding_tmps)[9+70] == 2);
	assert(((double*)coding_tmps)[9+71] == 1);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Linear nonlinearity, NoSign polarity split and SumAbs reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LINEAR;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NO_SIGN;
        enum reduce_type          reduce_type = SUM_ABS;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 1 + 1 + 2);
	assert(o_coeffs[1] == 3 + 3);
	assert(o_coeffs[2] == 2 + 3);
	assert(o_coeffs[3] == 1 + 1 + 2);
	assert(o_coeffs[4] == 2 + 3 + 3);
	assert(o_coeffs[5] == 2 + 3);
	assert(o_coeffs[6] == 1 + 2);
	assert(o_coeffs[7] == 3 + 3);
	assert(o_coeffs[8] == 2 + 3);
	assert(o_coeffs[9] == 2);
	assert(o_coeffs[10] == 3 + 3);
	assert(o_coeffs[11] == 2 + 3);
	assert(o_coeffs[12] == 1 + 2);
	assert(o_coeffs[13] == 2 + 3 + 3);
	assert(o_coeffs[14] == 2 + 3);
	assert(o_coeffs[15] == 2);
	assert(o_coeffs[16] == 3 + 3);
	assert(o_coeffs[17] == 2 + 3);
	assert(o_coeffs[18] == 1 + 1);
	assert(o_coeffs[19] == 1 + 1 + 1 + 2);
	assert(o_coeffs[20] == 3 + 3);
	assert(o_coeffs[21] == 1 + 1 + 3 + 3);
	assert(o_coeffs[22] == 1 + 1);
	assert(o_coeffs[23] == 1 + 1 + 1 + 2);
	assert(o_coeffs[24] == 1 + 2);
	assert(o_coeffs[25] == 2);
	assert(o_coeffs[26] == 3 + 3);
	assert(o_coeffs[27] == 3 + 3);
	assert(o_coeffs[28] == 1 + 2);
	assert(o_coeffs[29] == 2);
	assert(o_coeffs[30] == 1 + 2);
	assert(o_coeffs[31] == 3 + 3);
	assert(o_coeffs[32] == 1 + 2);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == 1);
	assert(((double*)coding_tmps)[9+2] == 2);
	assert(((double*)coding_tmps)[9+3] == 1);
	assert(((double*)coding_tmps)[9+4] == 1);
	assert(((double*)coding_tmps)[9+5] == 2);
	assert(((double*)coding_tmps)[9+6] == 2);
	assert(((double*)coding_tmps)[9+7] == 2);
	assert(((double*)coding_tmps)[9+8] == 3);
	assert(((double*)coding_tmps)[9+9] == 1);
	assert(((double*)coding_tmps)[9+10] == 3);
	assert(((double*)coding_tmps)[9+11] == 1);
	assert(((double*)coding_tmps)[9+12] == 3);
	assert(((double*)coding_tmps)[9+13] == 2);
	assert(((double*)coding_tmps)[9+14] == 3);
	assert(((double*)coding_tmps)[9+15] == 2);
	assert(((double*)coding_tmps)[9+16] == 3);
	assert(((double*)coding_tmps)[9+17] == 1);
	assert(((double*)coding_tmps)[9+18] == 2);
	assert(((double*)coding_tmps)[9+19] == 1);
	assert(((double*)coding_tmps)[9+20] == 3);
	assert(((double*)coding_tmps)[9+21] == 2);
	assert(((double*)coding_tmps)[9+22] == 2);
	assert(((double*)coding_tmps)[9+23] == 1);
	assert(((double*)coding_tmps)[9+24] == 1);
	assert(((double*)coding_tmps)[9+25] == 3);
	assert(((double*)coding_tmps)[9+26] == 2);
	assert(((double*)coding_tmps)[9+27] == 3);
	assert(((double*)coding_tmps)[9+28] == 1);
	assert(((double*)coding_tmps)[9+29] == 3);
	assert(((double*)coding_tmps)[9+30] == 2);
	assert(((double*)coding_tmps)[9+31] == 3);
	assert(((double*)coding_tmps)[9+32] == 3);
	assert(((double*)coding_tmps)[9+33] == 3);
	assert(((double*)coding_tmps)[9+34] == 3);
	assert(((double*)coding_tmps)[9+35] == 3);
	assert(((double*)coding_tmps)[9+36] == 3);
	assert(((double*)coding_tmps)[9+37] == 3);
	assert(((double*)coding_tmps)[9+38] == 3);
	assert(((double*)coding_tmps)[9+39] == 3);
	assert(((double*)coding_tmps)[9+40] == 3);
	assert(((double*)coding_tmps)[9+41] == 3);
	assert(((double*)coding_tmps)[9+42] == 2);
	assert(((double*)coding_tmps)[9+43] == 1);
	assert(((double*)coding_tmps)[9+44] == 3);
	assert(((double*)coding_tmps)[9+45] == 3);
	assert(((double*)coding_tmps)[9+46] == 2);
	assert(((double*)coding_tmps)[9+47] == 1);
	assert(((double*)coding_tmps)[9+48] == 1);
	assert(((double*)coding_tmps)[9+49] == 2);
	assert(((double*)coding_tmps)[9+50] == 2);
	assert(((double*)coding_tmps)[9+51] == 2);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == 1);
	assert(((double*)coding_tmps)[9+54] == 2);
	assert(((double*)coding_tmps)[9+55] == 1);
	assert(((double*)coding_tmps)[9+56] == 2);
	assert(((double*)coding_tmps)[9+57] == 3);
	assert(((double*)coding_tmps)[9+58] == 3);
	assert(((double*)coding_tmps)[9+59] == 2);
	assert(((double*)coding_tmps)[9+60] == 3);
	assert(((double*)coding_tmps)[9+61] == 1);
	assert(((double*)coding_tmps)[9+62] == 3);
	assert(((double*)coding_tmps)[9+63] == 1);
	assert(((double*)coding_tmps)[9+64] == 3);
	assert(((double*)coding_tmps)[9+65] == 2);
	assert(((double*)coding_tmps)[9+66] == 2);
	assert(((double*)coding_tmps)[9+67] == 1);
       	assert(((double*)coding_tmps)[9+68] == 3);
	assert(((double*)coding_tmps)[9+69] == 1);
	assert(((double*)coding_tmps)[9+70] == 2);
	assert(((double*)coding_tmps)[9+71] == 1);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Linear nonlinearity, NoSign polarity split and SumSqr reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LINEAR;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NO_SIGN;
        enum reduce_type          reduce_type = SUM_SQR;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 1*1 + 1*1 + 2*2);
	assert(o_coeffs[1] == 3*3 + 3*3);
	assert(o_coeffs[2] == 3*3 + 2*2);
	assert(o_coeffs[3] == 1*1 + 1*1 + 2*2);
	assert(o_coeffs[4] == 2*2 + 3*3 + 3*3);
	assert(o_coeffs[5] == 2*2 + 3*3);
	assert(o_coeffs[6] == 1*1 + 2*2);
	assert(o_coeffs[7] == 3*3 + 3*3);
	assert(o_coeffs[8] == 2*2 + 3*3);
	assert(o_coeffs[9] == 2*2);
	assert(o_coeffs[10] == 3*3 + 3*3);
	assert(o_coeffs[11] == 2*2 + 3*3);
	assert(o_coeffs[12] == 1*1 + 2*2);
	assert(o_coeffs[13] == 2*2 + 3*3 + 3*3);
	assert(o_coeffs[14] == 2*2 + 3*3);
	assert(o_coeffs[15] == 2*2);
	assert(o_coeffs[16] == 3*3 + 3*3);
	assert(o_coeffs[17] == 2*2 + 3*3);
	assert(o_coeffs[18] == 1*1 + 1*1);
	assert(o_coeffs[19] == 1*1 + 1*1 + 1*1 + 2*2);
	assert(o_coeffs[20] == 3*3 + 3*3);
	assert(o_coeffs[21] == 1*1 + 1*1 + 3*3 + 3*3);
	assert(o_coeffs[22] == 1*1 + 1*1);
	assert(o_coeffs[23] == 1*1 + 1*1 + 1*1 + 2*2);
	assert(o_coeffs[24] == 1*1 + 2*2);
	assert(o_coeffs[25] == 2*2);
	assert(o_coeffs[26] == 3*3 + 3*3);
	assert(o_coeffs[27] == 3*3 + 3*3);
	assert(o_coeffs[28] == 1*1 + 2*2);
	assert(o_coeffs[29] == 2*2);
	assert(o_coeffs[30] == 1*1 + 2*2);
	assert(o_coeffs[31] == 3*3 + 3*3);
	assert(o_coeffs[32] == 1*1 + 2*2);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == 1);
	assert(((double*)coding_tmps)[9+2] == 2);
	assert(((double*)coding_tmps)[9+3] == 1);
	assert(((double*)coding_tmps)[9+4] == 1);
	assert(((double*)coding_tmps)[9+5] == 2);
	assert(((double*)coding_tmps)[9+6] == 2);
	assert(((double*)coding_tmps)[9+7] == 2);
	assert(((double*)coding_tmps)[9+8] == 3);
	assert(((double*)coding_tmps)[9+9] == 1);
	assert(((double*)coding_tmps)[9+10] == 3);
	assert(((double*)coding_tmps)[9+11] == 1);
	assert(((double*)coding_tmps)[9+12] == 3);
	assert(((double*)coding_tmps)[9+13] == 2);
	assert(((double*)coding_tmps)[9+14] == 3);
	assert(((double*)coding_tmps)[9+15] == 2);
	assert(((double*)coding_tmps)[9+16] == 3);
	assert(((double*)coding_tmps)[9+17] == 1);
	assert(((double*)coding_tmps)[9+18] == 2);
	assert(((double*)coding_tmps)[9+19] == 1);
	assert(((double*)coding_tmps)[9+20] == 3);
	assert(((double*)coding_tmps)[9+21] == 2);
	assert(((double*)coding_tmps)[9+22] == 2);
	assert(((double*)coding_tmps)[9+23] == 1);
	assert(((double*)coding_tmps)[9+24] == 1);
	assert(((double*)coding_tmps)[9+25] == 3);
	assert(((double*)coding_tmps)[9+26] == 2);
	assert(((double*)coding_tmps)[9+27] == 3);
	assert(((double*)coding_tmps)[9+28] == 1);
	assert(((double*)coding_tmps)[9+29] == 3);
	assert(((double*)coding_tmps)[9+30] == 2);
	assert(((double*)coding_tmps)[9+31] == 3);
	assert(((double*)coding_tmps)[9+32] == 3);
	assert(((double*)coding_tmps)[9+33] == 3);
	assert(((double*)coding_tmps)[9+34] == 3);
	assert(((double*)coding_tmps)[9+35] == 3);
	assert(((double*)coding_tmps)[9+36] == 3);
	assert(((double*)coding_tmps)[9+37] == 3);
	assert(((double*)coding_tmps)[9+38] == 3);
	assert(((double*)coding_tmps)[9+39] == 3);
	assert(((double*)coding_tmps)[9+40] == 3);
	assert(((double*)coding_tmps)[9+41] == 3);
	assert(((double*)coding_tmps)[9+42] == 2);
	assert(((double*)coding_tmps)[9+43] == 1);
	assert(((double*)coding_tmps)[9+44] == 3);
	assert(((double*)coding_tmps)[9+45] == 3);
	assert(((double*)coding_tmps)[9+46] == 2);
	assert(((double*)coding_tmps)[9+47] == 1);
	assert(((double*)coding_tmps)[9+48] == 1);
	assert(((double*)coding_tmps)[9+49] == 2);
	assert(((double*)coding_tmps)[9+50] == 2);
	assert(((double*)coding_tmps)[9+51] == 2);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == 1);
	assert(((double*)coding_tmps)[9+54] == 2);
	assert(((double*)coding_tmps)[9+55] == 1);
	assert(((double*)coding_tmps)[9+56] == 2);
	assert(((double*)coding_tmps)[9+57] == 3);
	assert(((double*)coding_tmps)[9+58] == 3);
	assert(((double*)coding_tmps)[9+59] == 2);
	assert(((double*)coding_tmps)[9+60] == 3);
	assert(((double*)coding_tmps)[9+61] == 1);
	assert(((double*)coding_tmps)[9+62] == 3);
	assert(((double*)coding_tmps)[9+63] == 1);
	assert(((double*)coding_tmps)[9+64] == 3);
	assert(((double*)coding_tmps)[9+65] == 2);
	assert(((double*)coding_tmps)[9+66] == 2);
	assert(((double*)coding_tmps)[9+67] == 1);
       	assert(((double*)coding_tmps)[9+68] == 3);
	assert(((double*)coding_tmps)[9+69] == 1);
	assert(((double*)coding_tmps)[9+70] == 2);
	assert(((double*)coding_tmps)[9+71] == 1);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Linear nonlinearity, KeepSign polarity split and Subsample reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LINEAR;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = KEEP_SIGN;
        enum reduce_type          reduce_type = SUBSAMPLE;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 18);
	assert(o_coeffs[0] == 1);
	assert(o_coeffs[1] == 3);
	assert(o_coeffs[2] == 3);
	assert(o_coeffs[3] == 1);
	assert(o_coeffs[4] == 2);
	assert(o_coeffs[5] == 3);
	assert(o_coeffs[6] == 3);
	assert(o_coeffs[7] == 1);
	assert(o_coeffs[8] == 3);
	assert(o_coeffs[9] == 3);
	assert(o_coeffs[10] == -1);
	assert(o_coeffs[11] == -1);
	assert(o_coeffs[12] == -3);
	assert(o_coeffs[13] == -3);
	assert(o_coeffs[14] == -2);
	assert(o_coeffs[15] == -1);
	assert(o_coeffs[16] == -3);
	assert(o_coeffs[17] == -2);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 16);
	assert(o_coeffs_idx[6] == 17);
	assert(o_coeffs_idx[7] == 18);
	assert(o_coeffs_idx[8] == 19);
	assert(o_coeffs_idx[9] == 20);
	assert(o_coeffs_idx[10] == 54+28);
	assert(o_coeffs_idx[11] == 54+29);
	assert(o_coeffs_idx[12] == 54+31);
	assert(o_coeffs_idx[13] == 54+32);
	assert(o_coeffs_idx[14] == 54+35);
	assert(o_coeffs_idx[15] == 54+45);
	assert(o_coeffs_idx[16] == 54+48);
	assert(o_coeffs_idx[17] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -1);
	assert(((double*)coding_tmps)[9+2] == 2);
	assert(((double*)coding_tmps)[9+3] == -1);
	assert(((double*)coding_tmps)[9+4] == 1);
	assert(((double*)coding_tmps)[9+5] == -2);
	assert(((double*)coding_tmps)[9+6] == 2);
	assert(((double*)coding_tmps)[9+7] == -2);
	assert(((double*)coding_tmps)[9+8] == 3);
	assert(((double*)coding_tmps)[9+9] == -1);
	assert(((double*)coding_tmps)[9+10] == 3);
	assert(((double*)coding_tmps)[9+11] == -1);
	assert(((double*)coding_tmps)[9+12] == 3);
	assert(((double*)coding_tmps)[9+13] == 2);
	assert(((double*)coding_tmps)[9+14] == 3);
	assert(((double*)coding_tmps)[9+15] == -2);
	assert(((double*)coding_tmps)[9+16] == 3);
	assert(((double*)coding_tmps)[9+17] == -1);
	assert(((double*)coding_tmps)[9+18] == 2);
	assert(((double*)coding_tmps)[9+19] == -1);
	assert(((double*)coding_tmps)[9+20] == 3);
	assert(((double*)coding_tmps)[9+21] == -2);
	assert(((double*)coding_tmps)[9+22] == 2);
	assert(((double*)coding_tmps)[9+23] == -1);
	assert(((double*)coding_tmps)[9+24] == 1);
	assert(((double*)coding_tmps)[9+25] == -3);
	assert(((double*)coding_tmps)[9+26] == 2);
	assert(((double*)coding_tmps)[9+27] == -3);
	assert(((double*)coding_tmps)[9+28] == 1);
	assert(((double*)coding_tmps)[9+29] == -3);
	assert(((double*)coding_tmps)[9+30] == 2);
	assert(((double*)coding_tmps)[9+31] == -3);
	assert(((double*)coding_tmps)[9+32] == 3);
	assert(((double*)coding_tmps)[9+33] == -3);
	assert(((double*)coding_tmps)[9+34] == 3);
	assert(((double*)coding_tmps)[9+35] == -3);
	assert(((double*)coding_tmps)[9+36] == 3);
	assert(((double*)coding_tmps)[9+37] == -3);
	assert(((double*)coding_tmps)[9+38] == 3);
	assert(((double*)coding_tmps)[9+39] == -3);
	assert(((double*)coding_tmps)[9+40] == 3);
	assert(((double*)coding_tmps)[9+41] == -3);
	assert(((double*)coding_tmps)[9+42] == 2);
	assert(((double*)coding_tmps)[9+43] == -1);
	assert(((double*)coding_tmps)[9+44] == 3);
	assert(((double*)coding_tmps)[9+45] == -3);
	assert(((double*)coding_tmps)[9+46] == 2);
	assert(((double*)coding_tmps)[9+47] == -1);
	assert(((double*)coding_tmps)[9+48] == 1);
	assert(((double*)coding_tmps)[9+49] == -2);
	assert(((double*)coding_tmps)[9+50] == 2);
	assert(((double*)coding_tmps)[9+51] == -2);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -1);
	assert(((double*)coding_tmps)[9+54] == 2);
	assert(((double*)coding_tmps)[9+55] == -1);
	assert(((double*)coding_tmps)[9+56] == 2);
	assert(((double*)coding_tmps)[9+57] == 3);
	assert(((double*)coding_tmps)[9+58] == 3);
	assert(((double*)coding_tmps)[9+59] == -2);
	assert(((double*)coding_tmps)[9+60] == 3);
	assert(((double*)coding_tmps)[9+61] == -1);
	assert(((double*)coding_tmps)[9+62] == 3);
	assert(((double*)coding_tmps)[9+63] == -1);
	assert(((double*)coding_tmps)[9+64] == 3);
	assert(((double*)coding_tmps)[9+65] == -2);
	assert(((double*)coding_tmps)[9+66] == 2);
	assert(((double*)coding_tmps)[9+67] == -1);
       	assert(((double*)coding_tmps)[9+68] == 3);
	assert(((double*)coding_tmps)[9+69] == -1);
	assert(((double*)coding_tmps)[9+70] == 2);
	assert(((double*)coding_tmps)[9+71] == -1);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 0);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Linear nonlinearity, KeepSign polarity split and MaxNoSign reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LINEAR;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = KEEP_SIGN;
        enum reduce_type          reduce_type = MAX_NO_SIGN;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 2);
	assert(o_coeffs[1] == 3);
	assert(o_coeffs[2] == 3);
	assert(o_coeffs[3] == 2);
	assert(o_coeffs[4] == 3);
	assert(o_coeffs[5] == 3);
	assert(o_coeffs[6] == 2);
	assert(o_coeffs[7] == 3);
	assert(o_coeffs[8] == 3);
	assert(o_coeffs[9] == 2);
	assert(o_coeffs[10] == 3);
	assert(o_coeffs[11] == 3);
	assert(o_coeffs[12] == 2);
	assert(o_coeffs[13] == 3);
	assert(o_coeffs[14] == 3);
	assert(o_coeffs[15] == 2);
	assert(o_coeffs[16] == 3);
	assert(o_coeffs[17] == 3);
	assert(o_coeffs[18] == 1);
	assert(o_coeffs[19] == 2);
	assert(o_coeffs[20] == 3);
	assert(o_coeffs[21] == 3);
	assert(o_coeffs[22] == 1);
	assert(o_coeffs[23] == 2);
	assert(o_coeffs[24] == 2);
	assert(o_coeffs[25] == 2);
	assert(o_coeffs[26] == 3);
	assert(o_coeffs[27] == 3);
	assert(o_coeffs[28] == 2);
	assert(o_coeffs[29] == 2);
	assert(o_coeffs[30] == 2);
	assert(o_coeffs[31] == 3);
	assert(o_coeffs[32] == 2);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -1);
	assert(((double*)coding_tmps)[9+2] == 2);
	assert(((double*)coding_tmps)[9+3] == -1);
	assert(((double*)coding_tmps)[9+4] == 1);
	assert(((double*)coding_tmps)[9+5] == -2);
	assert(((double*)coding_tmps)[9+6] == 2);
	assert(((double*)coding_tmps)[9+7] == -2);
	assert(((double*)coding_tmps)[9+8] == 3);
	assert(((double*)coding_tmps)[9+9] == -1);
	assert(((double*)coding_tmps)[9+10] == 3);
	assert(((double*)coding_tmps)[9+11] == -1);
	assert(((double*)coding_tmps)[9+12] == 3);
	assert(((double*)coding_tmps)[9+13] == 2);
	assert(((double*)coding_tmps)[9+14] == 3);
	assert(((double*)coding_tmps)[9+15] == -2);
	assert(((double*)coding_tmps)[9+16] == 3);
	assert(((double*)coding_tmps)[9+17] == -1);
	assert(((double*)coding_tmps)[9+18] == 2);
	assert(((double*)coding_tmps)[9+19] == -1);
	assert(((double*)coding_tmps)[9+20] == 3);
	assert(((double*)coding_tmps)[9+21] == -2);
	assert(((double*)coding_tmps)[9+22] == 2);
	assert(((double*)coding_tmps)[9+23] == -1);
	assert(((double*)coding_tmps)[9+24] == 1);
	assert(((double*)coding_tmps)[9+25] == -3);
	assert(((double*)coding_tmps)[9+26] == 2);
	assert(((double*)coding_tmps)[9+27] == -3);
	assert(((double*)coding_tmps)[9+28] == 1);
	assert(((double*)coding_tmps)[9+29] == -3);
	assert(((double*)coding_tmps)[9+30] == 2);
	assert(((double*)coding_tmps)[9+31] == -3);
	assert(((double*)coding_tmps)[9+32] == 3);
	assert(((double*)coding_tmps)[9+33] == -3);
	assert(((double*)coding_tmps)[9+34] == 3);
	assert(((double*)coding_tmps)[9+35] == -3);
	assert(((double*)coding_tmps)[9+36] == 3);
	assert(((double*)coding_tmps)[9+37] == -3);
	assert(((double*)coding_tmps)[9+38] == 3);
	assert(((double*)coding_tmps)[9+39] == -3);
	assert(((double*)coding_tmps)[9+40] == 3);
	assert(((double*)coding_tmps)[9+41] == -3);
	assert(((double*)coding_tmps)[9+42] == 2);
	assert(((double*)coding_tmps)[9+43] == -1);
	assert(((double*)coding_tmps)[9+44] == 3);
	assert(((double*)coding_tmps)[9+45] == -3);
	assert(((double*)coding_tmps)[9+46] == 2);
	assert(((double*)coding_tmps)[9+47] == -1);
	assert(((double*)coding_tmps)[9+48] == 1);
	assert(((double*)coding_tmps)[9+49] == -2);
	assert(((double*)coding_tmps)[9+50] == 2);
	assert(((double*)coding_tmps)[9+51] == -2);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -1);
	assert(((double*)coding_tmps)[9+54] == 2);
	assert(((double*)coding_tmps)[9+55] == -1);
	assert(((double*)coding_tmps)[9+56] == 2);
	assert(((double*)coding_tmps)[9+57] == 3);
	assert(((double*)coding_tmps)[9+58] == 3);
	assert(((double*)coding_tmps)[9+59] == -2);
	assert(((double*)coding_tmps)[9+60] == 3);
	assert(((double*)coding_tmps)[9+61] == -1);
	assert(((double*)coding_tmps)[9+62] == 3);
	assert(((double*)coding_tmps)[9+63] == -1);
	assert(((double*)coding_tmps)[9+64] == 3);
	assert(((double*)coding_tmps)[9+65] == -2);
	assert(((double*)coding_tmps)[9+66] == 2);
	assert(((double*)coding_tmps)[9+67] == -1);
       	assert(((double*)coding_tmps)[9+68] == 3);
	assert(((double*)coding_tmps)[9+69] == -1);
	assert(((double*)coding_tmps)[9+70] == 2);
	assert(((double*)coding_tmps)[9+71] == -1);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Linear nonlinearity, KeepSign polarity split and MaxKeepSign reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LINEAR;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = KEEP_SIGN;
        enum reduce_type          reduce_type = MAX_KEEP_SIGN;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 2);
	assert(o_coeffs[1] == 3);
	assert(o_coeffs[2] == 3);
	assert(o_coeffs[3] == 2);
	assert(o_coeffs[4] == 3);
	assert(o_coeffs[5] == 3);
	assert(o_coeffs[6] == 2);
	assert(o_coeffs[7] == 3);
	assert(o_coeffs[8] == 3);
	assert(o_coeffs[9] == 2);
	assert(o_coeffs[10] == 3);
	assert(o_coeffs[11] == 3);
	assert(o_coeffs[12] == 2);
	assert(o_coeffs[13] == 3);
	assert(o_coeffs[14] == 3);
	assert(o_coeffs[15] == 2);
	assert(o_coeffs[16] == 3);
	assert(o_coeffs[17] == 3);
	assert(o_coeffs[18] == -1);
	assert(o_coeffs[19] == -2);
	assert(o_coeffs[20] == -3);
	assert(o_coeffs[21] == -3);
	assert(o_coeffs[22] == -1);
	assert(o_coeffs[23] == -2);
	assert(o_coeffs[24] == -2);
	assert(o_coeffs[25] == -2);
	assert(o_coeffs[26] == -3);
	assert(o_coeffs[27] == -3);
	assert(o_coeffs[28] == -2);
	assert(o_coeffs[29] == -2);
	assert(o_coeffs[30] == -2);
	assert(o_coeffs[31] == -3);
	assert(o_coeffs[32] == -2);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -1);
	assert(((double*)coding_tmps)[9+2] == 2);
	assert(((double*)coding_tmps)[9+3] == -1);
	assert(((double*)coding_tmps)[9+4] == 1);
	assert(((double*)coding_tmps)[9+5] == -2);
	assert(((double*)coding_tmps)[9+6] == 2);
	assert(((double*)coding_tmps)[9+7] == -2);
	assert(((double*)coding_tmps)[9+8] == 3);
	assert(((double*)coding_tmps)[9+9] == -1);
	assert(((double*)coding_tmps)[9+10] == 3);
	assert(((double*)coding_tmps)[9+11] == -1);
	assert(((double*)coding_tmps)[9+12] == 3);
	assert(((double*)coding_tmps)[9+13] == 2);
	assert(((double*)coding_tmps)[9+14] == 3);
	assert(((double*)coding_tmps)[9+15] == -2);
	assert(((double*)coding_tmps)[9+16] == 3);
	assert(((double*)coding_tmps)[9+17] == -1);
	assert(((double*)coding_tmps)[9+18] == 2);
	assert(((double*)coding_tmps)[9+19] == -1);
	assert(((double*)coding_tmps)[9+20] == 3);
	assert(((double*)coding_tmps)[9+21] == -2);
	assert(((double*)coding_tmps)[9+22] == 2);
	assert(((double*)coding_tmps)[9+23] == -1);
	assert(((double*)coding_tmps)[9+24] == 1);
	assert(((double*)coding_tmps)[9+25] == -3);
	assert(((double*)coding_tmps)[9+26] == 2);
	assert(((double*)coding_tmps)[9+27] == -3);
	assert(((double*)coding_tmps)[9+28] == 1);
	assert(((double*)coding_tmps)[9+29] == -3);
	assert(((double*)coding_tmps)[9+30] == 2);
	assert(((double*)coding_tmps)[9+31] == -3);
	assert(((double*)coding_tmps)[9+32] == 3);
	assert(((double*)coding_tmps)[9+33] == -3);
	assert(((double*)coding_tmps)[9+34] == 3);
	assert(((double*)coding_tmps)[9+35] == -3);
	assert(((double*)coding_tmps)[9+36] == 3);
	assert(((double*)coding_tmps)[9+37] == -3);
	assert(((double*)coding_tmps)[9+38] == 3);
	assert(((double*)coding_tmps)[9+39] == -3);
	assert(((double*)coding_tmps)[9+40] == 3);
	assert(((double*)coding_tmps)[9+41] == -3);
	assert(((double*)coding_tmps)[9+42] == 2);
	assert(((double*)coding_tmps)[9+43] == -1);
	assert(((double*)coding_tmps)[9+44] == 3);
	assert(((double*)coding_tmps)[9+45] == -3);
	assert(((double*)coding_tmps)[9+46] == 2);
	assert(((double*)coding_tmps)[9+47] == -1);
	assert(((double*)coding_tmps)[9+48] == 1);
	assert(((double*)coding_tmps)[9+49] == -2);
	assert(((double*)coding_tmps)[9+50] == 2);
	assert(((double*)coding_tmps)[9+51] == -2);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -1);
	assert(((double*)coding_tmps)[9+54] == 2);
	assert(((double*)coding_tmps)[9+55] == -1);
	assert(((double*)coding_tmps)[9+56] == 2);
	assert(((double*)coding_tmps)[9+57] == 3);
	assert(((double*)coding_tmps)[9+58] == 3);
	assert(((double*)coding_tmps)[9+59] == -2);
	assert(((double*)coding_tmps)[9+60] == 3);
	assert(((double*)coding_tmps)[9+61] == -1);
	assert(((double*)coding_tmps)[9+62] == 3);
	assert(((double*)coding_tmps)[9+63] == -1);
	assert(((double*)coding_tmps)[9+64] == 3);
	assert(((double*)coding_tmps)[9+65] == -2);
	assert(((double*)coding_tmps)[9+66] == 2);
	assert(((double*)coding_tmps)[9+67] == -1);
       	assert(((double*)coding_tmps)[9+68] == 3);
	assert(((double*)coding_tmps)[9+69] == -1);
	assert(((double*)coding_tmps)[9+70] == 2);
	assert(((double*)coding_tmps)[9+71] == -1);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Linear nonlinearity, KeepSign polarity split and SumAbs reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LINEAR;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = KEEP_SIGN;
        enum reduce_type          reduce_type = SUM_ABS;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 1 + 1 + 2);
	assert(o_coeffs[1] == 3 + 3);
	assert(o_coeffs[2] == 2 + 3);
	assert(o_coeffs[3] == 1 + 1 + 2);
	assert(o_coeffs[4] == 2 + 3 + 3);
	assert(o_coeffs[5] == 2 + 3);
	assert(o_coeffs[6] == 1 + 2);
	assert(o_coeffs[7] == 3 + 3);
	assert(o_coeffs[8] == 2 + 3);
	assert(o_coeffs[9] == 2);
	assert(o_coeffs[10] == 3 + 3);
	assert(o_coeffs[11] == 2 + 3);
	assert(o_coeffs[12] == 1 + 2);
	assert(o_coeffs[13] == 2 + 3 + 3);
	assert(o_coeffs[14] == 2 + 3);
	assert(o_coeffs[15] == 2);
	assert(o_coeffs[16] == 3 + 3);
	assert(o_coeffs[17] == 2 + 3);
	assert(o_coeffs[18] == 1 + 1);
	assert(o_coeffs[19] == 1 + 1 + 1 + 2);
	assert(o_coeffs[20] == 3 + 3);
	assert(o_coeffs[21] == 1 + 1 + 3 + 3);
	assert(o_coeffs[22] == 1 + 1);
	assert(o_coeffs[23] == 1 + 1 + 1 + 2);
	assert(o_coeffs[24] == 1 + 2);
	assert(o_coeffs[25] == 2);
	assert(o_coeffs[26] == 3 + 3);
	assert(o_coeffs[27] == 3 + 3);
	assert(o_coeffs[28] == 1 + 2);
	assert(o_coeffs[29] == 2);
	assert(o_coeffs[30] == 1 + 2);
	assert(o_coeffs[31] == 3 + 3);
	assert(o_coeffs[32] == 1 + 2);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -1);
	assert(((double*)coding_tmps)[9+2] == 2);
	assert(((double*)coding_tmps)[9+3] == -1);
	assert(((double*)coding_tmps)[9+4] == 1);
	assert(((double*)coding_tmps)[9+5] == -2);
	assert(((double*)coding_tmps)[9+6] == 2);
	assert(((double*)coding_tmps)[9+7] == -2);
	assert(((double*)coding_tmps)[9+8] == 3);
	assert(((double*)coding_tmps)[9+9] == -1);
	assert(((double*)coding_tmps)[9+10] == 3);
	assert(((double*)coding_tmps)[9+11] == -1);
	assert(((double*)coding_tmps)[9+12] == 3);
	assert(((double*)coding_tmps)[9+13] == 2);
	assert(((double*)coding_tmps)[9+14] == 3);
	assert(((double*)coding_tmps)[9+15] == -2);
	assert(((double*)coding_tmps)[9+16] == 3);
	assert(((double*)coding_tmps)[9+17] == -1);
	assert(((double*)coding_tmps)[9+18] == 2);
	assert(((double*)coding_tmps)[9+19] == -1);
	assert(((double*)coding_tmps)[9+20] == 3);
	assert(((double*)coding_tmps)[9+21] == -2);
	assert(((double*)coding_tmps)[9+22] == 2);
	assert(((double*)coding_tmps)[9+23] == -1);
	assert(((double*)coding_tmps)[9+24] == 1);
	assert(((double*)coding_tmps)[9+25] == -3);
	assert(((double*)coding_tmps)[9+26] == 2);
	assert(((double*)coding_tmps)[9+27] == -3);
	assert(((double*)coding_tmps)[9+28] == 1);
	assert(((double*)coding_tmps)[9+29] == -3);
	assert(((double*)coding_tmps)[9+30] == 2);
	assert(((double*)coding_tmps)[9+31] == -3);
	assert(((double*)coding_tmps)[9+32] == 3);
	assert(((double*)coding_tmps)[9+33] == -3);
	assert(((double*)coding_tmps)[9+34] == 3);
	assert(((double*)coding_tmps)[9+35] == -3);
	assert(((double*)coding_tmps)[9+36] == 3);
	assert(((double*)coding_tmps)[9+37] == -3);
	assert(((double*)coding_tmps)[9+38] == 3);
	assert(((double*)coding_tmps)[9+39] == -3);
	assert(((double*)coding_tmps)[9+40] == 3);
	assert(((double*)coding_tmps)[9+41] == -3);
	assert(((double*)coding_tmps)[9+42] == 2);
	assert(((double*)coding_tmps)[9+43] == -1);
	assert(((double*)coding_tmps)[9+44] == 3);
	assert(((double*)coding_tmps)[9+45] == -3);
	assert(((double*)coding_tmps)[9+46] == 2);
	assert(((double*)coding_tmps)[9+47] == -1);
	assert(((double*)coding_tmps)[9+48] == 1);
	assert(((double*)coding_tmps)[9+49] == -2);
	assert(((double*)coding_tmps)[9+50] == 2);
	assert(((double*)coding_tmps)[9+51] == -2);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -1);
	assert(((double*)coding_tmps)[9+54] == 2);
	assert(((double*)coding_tmps)[9+55] == -1);
	assert(((double*)coding_tmps)[9+56] == 2);
	assert(((double*)coding_tmps)[9+57] == 3);
	assert(((double*)coding_tmps)[9+58] == 3);
	assert(((double*)coding_tmps)[9+59] == -2);
	assert(((double*)coding_tmps)[9+60] == 3);
	assert(((double*)coding_tmps)[9+61] == -1);
	assert(((double*)coding_tmps)[9+62] == 3);
	assert(((double*)coding_tmps)[9+63] == -1);
	assert(((double*)coding_tmps)[9+64] == 3);
	assert(((double*)coding_tmps)[9+65] == -2);
	assert(((double*)coding_tmps)[9+66] == 2);
	assert(((double*)coding_tmps)[9+67] == -1);
       	assert(((double*)coding_tmps)[9+68] == 3);
	assert(((double*)coding_tmps)[9+69] == -1);
	assert(((double*)coding_tmps)[9+70] == 2);
	assert(((double*)coding_tmps)[9+71] == -1);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Linear nonlinearity, KeepSign polarity split and SumSqr reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LINEAR;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = KEEP_SIGN;
        enum reduce_type          reduce_type = SUM_SQR;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 1*1 + 1*1 + 2*2);
	assert(o_coeffs[1] == 3*3 + 3*3);
	assert(o_coeffs[2] == 3*3 + 2*2);
	assert(o_coeffs[3] == 1*1 + 1*1 + 2*2);
	assert(o_coeffs[4] == 2*2 + 3*3 + 3*3);
	assert(o_coeffs[5] == 2*2 + 3*3);
	assert(o_coeffs[6] == 1*1 + 2*2);
	assert(o_coeffs[7] == 3*3 + 3*3);
	assert(o_coeffs[8] == 2*2 + 3*3);
	assert(o_coeffs[9] == 2*2);
	assert(o_coeffs[10] == 3*3 + 3*3);
	assert(o_coeffs[11] == 2*2 + 3*3);
	assert(o_coeffs[12] == 1*1 + 2*2);
	assert(o_coeffs[13] == 2*2 + 3*3 + 3*3);
	assert(o_coeffs[14] == 2*2 + 3*3);
	assert(o_coeffs[15] == 2*2);
	assert(o_coeffs[16] == 3*3 + 3*3);
	assert(o_coeffs[17] == 2*2 + 3*3);
	assert(o_coeffs[18] == 1*1 + 1*1);
	assert(o_coeffs[19] == 1*1 + 1*1 + 1*1 + 2*2);
	assert(o_coeffs[20] == 3*3 + 3*3);
	assert(o_coeffs[21] == 1*1 + 1*1 + 3*3 + 3*3);
	assert(o_coeffs[22] == 1*1 + 1*1);
	assert(o_coeffs[23] == 1*1 + 1*1 + 1*1 + 2*2);
	assert(o_coeffs[24] == 1*1 + 2*2);
	assert(o_coeffs[25] == 2*2);
	assert(o_coeffs[26] == 3*3 + 3*3);
	assert(o_coeffs[27] == 3*3 + 3*3);
	assert(o_coeffs[28] == 1*1 + 2*2);
	assert(o_coeffs[29] == 2*2);
	assert(o_coeffs[30] == 1*1 + 2*2);
	assert(o_coeffs[31] == 3*3 + 3*3);
	assert(o_coeffs[32] == 1*1 + 2*2);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -1);
	assert(((double*)coding_tmps)[9+2] == 2);
	assert(((double*)coding_tmps)[9+3] == -1);
	assert(((double*)coding_tmps)[9+4] == 1);
	assert(((double*)coding_tmps)[9+5] == -2);
	assert(((double*)coding_tmps)[9+6] == 2);
	assert(((double*)coding_tmps)[9+7] == -2);
	assert(((double*)coding_tmps)[9+8] == 3);
	assert(((double*)coding_tmps)[9+9] == -1);
	assert(((double*)coding_tmps)[9+10] == 3);
	assert(((double*)coding_tmps)[9+11] == -1);
	assert(((double*)coding_tmps)[9+12] == 3);
	assert(((double*)coding_tmps)[9+13] == 2);
	assert(((double*)coding_tmps)[9+14] == 3);
	assert(((double*)coding_tmps)[9+15] == -2);
	assert(((double*)coding_tmps)[9+16] == 3);
	assert(((double*)coding_tmps)[9+17] == -1);
	assert(((double*)coding_tmps)[9+18] == 2);
	assert(((double*)coding_tmps)[9+19] == -1);
	assert(((double*)coding_tmps)[9+20] == 3);
	assert(((double*)coding_tmps)[9+21] == -2);
	assert(((double*)coding_tmps)[9+22] == 2);
	assert(((double*)coding_tmps)[9+23] == -1);
	assert(((double*)coding_tmps)[9+24] == 1);
	assert(((double*)coding_tmps)[9+25] == -3);
	assert(((double*)coding_tmps)[9+26] == 2);
	assert(((double*)coding_tmps)[9+27] == -3);
	assert(((double*)coding_tmps)[9+28] == 1);
	assert(((double*)coding_tmps)[9+29] == -3);
	assert(((double*)coding_tmps)[9+30] == 2);
	assert(((double*)coding_tmps)[9+31] == -3);
	assert(((double*)coding_tmps)[9+32] == 3);
	assert(((double*)coding_tmps)[9+33] == -3);
	assert(((double*)coding_tmps)[9+34] == 3);
	assert(((double*)coding_tmps)[9+35] == -3);
	assert(((double*)coding_tmps)[9+36] == 3);
	assert(((double*)coding_tmps)[9+37] == -3);
	assert(((double*)coding_tmps)[9+38] == 3);
	assert(((double*)coding_tmps)[9+39] == -3);
	assert(((double*)coding_tmps)[9+40] == 3);
	assert(((double*)coding_tmps)[9+41] == -3);
	assert(((double*)coding_tmps)[9+42] == 2);
	assert(((double*)coding_tmps)[9+43] == -1);
	assert(((double*)coding_tmps)[9+44] == 3);
	assert(((double*)coding_tmps)[9+45] == -3);
	assert(((double*)coding_tmps)[9+46] == 2);
	assert(((double*)coding_tmps)[9+47] == -1);
	assert(((double*)coding_tmps)[9+48] == 1);
	assert(((double*)coding_tmps)[9+49] == -2);
	assert(((double*)coding_tmps)[9+50] == 2);
	assert(((double*)coding_tmps)[9+51] == -2);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -1);
	assert(((double*)coding_tmps)[9+54] == 2);
	assert(((double*)coding_tmps)[9+55] == -1);
	assert(((double*)coding_tmps)[9+56] == 2);
	assert(((double*)coding_tmps)[9+57] == 3);
	assert(((double*)coding_tmps)[9+58] == 3);
	assert(((double*)coding_tmps)[9+59] == -2);
	assert(((double*)coding_tmps)[9+60] == 3);
	assert(((double*)coding_tmps)[9+61] == -1);
	assert(((double*)coding_tmps)[9+62] == 3);
	assert(((double*)coding_tmps)[9+63] == -1);
	assert(((double*)coding_tmps)[9+64] == 3);
	assert(((double*)coding_tmps)[9+65] == -2);
	assert(((double*)coding_tmps)[9+66] == 2);
	assert(((double*)coding_tmps)[9+67] == -1);
       	assert(((double*)coding_tmps)[9+68] == 3);
	assert(((double*)coding_tmps)[9+69] == -1);
	assert(((double*)coding_tmps)[9+70] == 2);
	assert(((double*)coding_tmps)[9+71] == -1);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Logistic nonlinearity, None polarity split and Subsample reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LOGISTIC;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NONE;
        enum reduce_type          reduce_type = SUBSAMPLE;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 18);
	assert(fabs(o_coeffs[0] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[1] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[2] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[3] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[4] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[5] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[6] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[7] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[8] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[9] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[10] - fast_logistic(-1)) < 1e-4);
	assert(fabs(o_coeffs[11] - fast_logistic(-1)) < 1e-4);
	assert(fabs(o_coeffs[12] - fast_logistic(-3)) < 1e-4);
	assert(fabs(o_coeffs[13] - fast_logistic(-3)) < 1e-4);
	assert(fabs(o_coeffs[14] - fast_logistic(-2)) < 1e-4);
	assert(fabs(o_coeffs[15] - fast_logistic(-1)) < 1e-4);
	assert(fabs(o_coeffs[16] - fast_logistic(-3)) < 1e-4);
	assert(fabs(o_coeffs[17] - fast_logistic(-2)) < 1e-4);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 16);
	assert(o_coeffs_idx[6] == 17);
	assert(o_coeffs_idx[7] == 18);
	assert(o_coeffs_idx[8] == 19);
	assert(o_coeffs_idx[9] == 20);
	assert(o_coeffs_idx[10] == 28);
	assert(o_coeffs_idx[11] == 29);
	assert(o_coeffs_idx[12] == 31);
	assert(o_coeffs_idx[13] == 32);
	assert(o_coeffs_idx[14] == 35);
	assert(o_coeffs_idx[15] == 45);
	assert(o_coeffs_idx[16] == 48);
	assert(o_coeffs_idx[17] == 51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(fabs(((double*)coding_tmps)[9+0] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+1] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+2] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+3] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+4] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+5] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+6] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+7] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+8] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+9] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+10] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+11] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+12] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+13] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+14] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+15] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+16] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+17] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+18] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+19] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+20] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+21] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+22] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+23] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+24] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+25] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+26] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+27] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+28] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+29] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+30] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+31] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+32] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+33] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+34] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+35] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+36] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+37] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+38] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+39] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+40] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+41] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+42] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+43] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+44] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+45] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+46] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+47] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+48] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+49] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+50] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+51] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+52] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+53] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+54] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+55] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+56] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+57] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+58] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+59] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+60] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+61] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+62] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+63] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+64] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+65] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+66] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+67] - fast_logistic(-1)) < 1e-4);
       	assert(fabs(((double*)coding_tmps)[9+68] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+69] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+70] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+71] - fast_logistic(-1)) < 1e-4);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 3);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 3);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 0);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Logistic nonlinearity, None polarity split and MaxNoSign reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LOGISTIC;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NONE;
        enum reduce_type          reduce_type = MAX_NO_SIGN;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(fabs(o_coeffs[0] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[1] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[2] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[3] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[4] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[5] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[6] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[7] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[8] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[9] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[10] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[11] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[12] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[13] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[14] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[15] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[16] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[17] - fast_logistic(3)) < 1e-4);
        assert(fabs(o_coeffs[18] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[19] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[20] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[21] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[22] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[23] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[24] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[25] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[26] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[27] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[28] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[29] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[30] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[31] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[32] - fast_logistic(2)) < 1e-4);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 28);
	assert(o_coeffs_idx[19] == 29);
	assert(o_coeffs_idx[20] == 31);
	assert(o_coeffs_idx[21] == 32);
	assert(o_coeffs_idx[22] == 34);
	assert(o_coeffs_idx[23] == 35);
	assert(o_coeffs_idx[24] == 36);
	assert(o_coeffs_idx[25] == 37);
	assert(o_coeffs_idx[26] == 39);
	assert(o_coeffs_idx[27] == 40);
	assert(o_coeffs_idx[28] == 42);
	assert(o_coeffs_idx[29] == 43);
	assert(o_coeffs_idx[30] == 45);
	assert(o_coeffs_idx[31] == 48);
	assert(o_coeffs_idx[32] == 51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(fabs(((double*)coding_tmps)[9+0] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+1] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+2] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+3] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+4] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+5] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+6] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+7] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+8] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+9] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+10] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+11] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+12] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+13] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+14] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+15] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+16] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+17] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+18] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+19] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+20] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+21] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+22] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+23] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+24] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+25] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+26] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+27] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+28] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+29] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+30] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+31] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+32] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+33] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+34] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+35] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+36] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+37] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+38] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+39] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+40] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+41] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+42] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+43] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+44] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+45] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+46] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+47] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+48] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+49] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+50] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+51] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+52] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+53] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+54] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+55] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+56] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+57] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+58] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+59] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+60] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+61] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+62] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+63] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+64] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+65] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+66] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+67] - fast_logistic(-1)) < 1e-4);
       	assert(fabs(((double*)coding_tmps)[9+68] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+69] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+70] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+71] - fast_logistic(-1)) < 1e-4);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 3);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 3);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Logistic nonlinearity, None polarity split and MaxKeepSign reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LOGISTIC;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NONE;
        enum reduce_type          reduce_type = MAX_KEEP_SIGN;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(fabs(o_coeffs[0] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[1] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[2] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[3] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[4] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[5] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[6] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[7] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[8] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[9] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[10] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[11] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[12] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[13] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[14] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[15] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[16] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[17] - fast_logistic(3)) < 1e-4);
        assert(fabs(o_coeffs[18] - fast_logistic(-1)) < 1e-4);
	assert(fabs(o_coeffs[19] - fast_logistic(-2)) < 1e-4);
	assert(fabs(o_coeffs[20] - fast_logistic(-3)) < 1e-4);
	assert(fabs(o_coeffs[21] - fast_logistic(-3)) < 1e-4);
	assert(fabs(o_coeffs[22] - fast_logistic(-1)) < 1e-4);
	assert(fabs(o_coeffs[23] - fast_logistic(-2)) < 1e-4);
	assert(fabs(o_coeffs[24] - fast_logistic(-2)) < 1e-4);
	assert(fabs(o_coeffs[25] - fast_logistic(-2)) < 1e-4);
	assert(fabs(o_coeffs[26] - fast_logistic(-3)) < 1e-4);
	assert(fabs(o_coeffs[27] - fast_logistic(-3)) < 1e-4);
	assert(fabs(o_coeffs[28] - fast_logistic(-2)) < 1e-4);
	assert(fabs(o_coeffs[29] - fast_logistic(-2)) < 1e-4);
	assert(fabs(o_coeffs[30] - fast_logistic(-2)) < 1e-4);
	assert(fabs(o_coeffs[31] - fast_logistic(-3)) < 1e-4);
	assert(fabs(o_coeffs[32] - fast_logistic(-2)) < 1e-4);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 28);
	assert(o_coeffs_idx[19] == 29);
	assert(o_coeffs_idx[20] == 31);
	assert(o_coeffs_idx[21] == 32);
	assert(o_coeffs_idx[22] == 34);
	assert(o_coeffs_idx[23] == 35);
	assert(o_coeffs_idx[24] == 36);
	assert(o_coeffs_idx[25] == 37);
	assert(o_coeffs_idx[26] == 39);
	assert(o_coeffs_idx[27] == 40);
	assert(o_coeffs_idx[28] == 42);
	assert(o_coeffs_idx[29] == 43);
	assert(o_coeffs_idx[30] == 45);
	assert(o_coeffs_idx[31] == 48);
	assert(o_coeffs_idx[32] == 51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(fabs(((double*)coding_tmps)[9+0] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+1] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+2] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+3] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+4] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+5] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+6] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+7] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+8] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+9] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+10] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+11] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+12] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+13] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+14] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+15] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+16] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+17] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+18] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+19] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+20] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+21] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+22] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+23] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+24] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+25] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+26] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+27] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+28] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+29] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+30] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+31] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+32] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+33] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+34] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+35] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+36] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+37] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+38] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+39] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+40] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+41] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+42] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+43] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+44] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+45] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+46] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+47] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+48] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+49] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+50] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+51] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+52] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+53] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+54] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+55] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+56] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+57] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+58] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+59] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+60] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+61] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+62] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+63] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+64] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+65] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+66] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+67] - fast_logistic(-1)) < 1e-4);
       	assert(fabs(((double*)coding_tmps)[9+68] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+69] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+70] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+71] - fast_logistic(-1)) < 1e-4);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 3);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 3);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Logistic nonlinearity, None polarity split and SumAbs reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LOGISTIC;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NONE;
        enum reduce_type          reduce_type = SUM_ABS;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(fabs(o_coeffs[0] - (fast_logistic(1) + fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[1] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[2] - (fast_logistic(2) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[3] - (fast_logistic(1) + fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[4] - (fast_logistic(2) + fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[5] - (fast_logistic(2) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[6] - (fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[7] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[8] - (fast_logistic(2) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[9] - (fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[10] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[11] - (fast_logistic(2) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[12] - (fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[13] - (fast_logistic(2) + fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[14] - (fast_logistic(2) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[15] - (fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[16] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[17] - (fast_logistic(2) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[18] - (fast_logistic(1) + fast_logistic(1))) < 1e-4);
	assert(fabs(o_coeffs[19] - (fast_logistic(1) + fast_logistic(1) + fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[20] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[21] - (fast_logistic(1) + fast_logistic(1) + fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[22] - (fast_logistic(1) + fast_logistic(1))) < 1e-4);
	assert(fabs(o_coeffs[23] - (fast_logistic(1) + fast_logistic(1) + fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[24] - (fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[25] - (fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[26] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[27] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[28] - (fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[29] - (fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[30] - (fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[31] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[32] - (fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 28);
	assert(o_coeffs_idx[19] == 29);
	assert(o_coeffs_idx[20] == 31);
	assert(o_coeffs_idx[21] == 32);
	assert(o_coeffs_idx[22] == 34);
	assert(o_coeffs_idx[23] == 35);
	assert(o_coeffs_idx[24] == 36);
	assert(o_coeffs_idx[25] == 37);
	assert(o_coeffs_idx[26] == 39);
	assert(o_coeffs_idx[27] == 40);
	assert(o_coeffs_idx[28] == 42);
	assert(o_coeffs_idx[29] == 43);
	assert(o_coeffs_idx[30] == 45);
	assert(o_coeffs_idx[31] == 48);
	assert(o_coeffs_idx[32] == 51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(fabs(((double*)coding_tmps)[9+0] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+1] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+2] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+3] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+4] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+5] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+6] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+7] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+8] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+9] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+10] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+11] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+12] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+13] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+14] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+15] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+16] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+17] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+18] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+19] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+20] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+21] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+22] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+23] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+24] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+25] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+26] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+27] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+28] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+29] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+30] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+31] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+32] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+33] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+34] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+35] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+36] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+37] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+38] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+39] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+40] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+41] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+42] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+43] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+44] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+45] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+46] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+47] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+48] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+49] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+50] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+51] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+52] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+53] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+54] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+55] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+56] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+57] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+58] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+59] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+60] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+61] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+62] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+63] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+64] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+65] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+66] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+67] - fast_logistic(-1)) < 1e-4);
       	assert(fabs(((double*)coding_tmps)[9+68] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+69] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+70] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+71] - fast_logistic(-1)) < 1e-4);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 3);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 3);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Logistic nonlinearity, None polarity split and SumSqr reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LOGISTIC;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NONE;
        enum reduce_type          reduce_type = SUM_SQR;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(fabs(o_coeffs[0] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[1] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[2] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[3] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[4] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[5] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[6] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[7] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[8] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[9] - (fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[10] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[11] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[12] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[13] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[14] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[15] - (fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[16] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[17] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[18] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1))) < 1e-4);
	assert(fabs(o_coeffs[19] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[20] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[21] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[22] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1))) < 1e-4);
	assert(fabs(o_coeffs[23] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[24] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[25] - (fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[26] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[27] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[28] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[29] - (fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[30] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[31] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[32] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 28);
	assert(o_coeffs_idx[19] == 29);
	assert(o_coeffs_idx[20] == 31);
	assert(o_coeffs_idx[21] == 32);
	assert(o_coeffs_idx[22] == 34);
	assert(o_coeffs_idx[23] == 35);
	assert(o_coeffs_idx[24] == 36);
	assert(o_coeffs_idx[25] == 37);
	assert(o_coeffs_idx[26] == 39);
	assert(o_coeffs_idx[27] == 40);
	assert(o_coeffs_idx[28] == 42);
	assert(o_coeffs_idx[29] == 43);
	assert(o_coeffs_idx[30] == 45);
	assert(o_coeffs_idx[31] == 48);
	assert(o_coeffs_idx[32] == 51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(fabs(((double*)coding_tmps)[9+0] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+1] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+2] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+3] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+4] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+5] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+6] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+7] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+8] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+9] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+10] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+11] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+12] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+13] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+14] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+15] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+16] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+17] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+18] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+19] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+20] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+21] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+22] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+23] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+24] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+25] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+26] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+27] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+28] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+29] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+30] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+31] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+32] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+33] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+34] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+35] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+36] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+37] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+38] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+39] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+40] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+41] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+42] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+43] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+44] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+45] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+46] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+47] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+48] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+49] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+50] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+51] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+52] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+53] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+54] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+55] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+56] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+57] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+58] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+59] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+60] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+61] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+62] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+63] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+64] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+65] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+66] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+67] - fast_logistic(-1)) < 1e-4);
       	assert(fabs(((double*)coding_tmps)[9+68] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+69] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+70] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+71] - fast_logistic(-1)) < 1e-4);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 3);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 3);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Logistic nonlinearity, NoSign polarity split and Subsample reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LOGISTIC;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NO_SIGN;
        enum reduce_type          reduce_type = SUBSAMPLE;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 18);
	assert(fabs(o_coeffs[0] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[1] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[2] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[3] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[4] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[5] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[6] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[7] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[8] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[9] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[10] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[11] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[12] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[13] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[14] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[15] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[16] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[17] - fast_logistic(2)) < 1e-4);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 16);
	assert(o_coeffs_idx[6] == 17);
	assert(o_coeffs_idx[7] == 18);
	assert(o_coeffs_idx[8] == 19);
	assert(o_coeffs_idx[9] == 20);
	assert(o_coeffs_idx[10] == 54+28);
	assert(o_coeffs_idx[11] == 54+29);
	assert(o_coeffs_idx[12] == 54+31);
	assert(o_coeffs_idx[13] == 54+32);
	assert(o_coeffs_idx[14] == 54+35);
	assert(o_coeffs_idx[15] == 54+45);
	assert(o_coeffs_idx[16] == 54+48);
	assert(o_coeffs_idx[17] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(fabs(((double*)coding_tmps)[9+0] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+1] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+2] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+3] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+4] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+5] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+6] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+7] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+8] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+9] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+10] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+11] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+12] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+13] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+14] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+15] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+16] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+17] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+18] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+19] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+20] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+21] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+22] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+23] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+24] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+25] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+26] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+27] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+28] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+29] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+30] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+31] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+32] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+33] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+34] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+35] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+36] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+37] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+38] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+39] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+40] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+41] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+42] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+43] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+44] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+45] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+46] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+47] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+48] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+49] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+50] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+51] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+52] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+53] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+54] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+55] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+56] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+57] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+58] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+59] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+60] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+61] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+62] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+63] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+64] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+65] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+66] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+67] - fast_logistic(1)) < 1e-4);
       	assert(fabs(((double*)coding_tmps)[9+68] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+69] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+70] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+71] - fast_logistic(1)) < 1e-4);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 0);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Logistic nonlinearity, NoSign polarity split and MaxNoSign reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LOGISTIC;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NO_SIGN;
        enum reduce_type          reduce_type = MAX_NO_SIGN;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(fabs(o_coeffs[0] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[1] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[2] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[3] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[4] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[5] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[6] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[7] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[8] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[9] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[10] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[11] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[12] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[13] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[14] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[15] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[16] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[17] - fast_logistic(3)) < 1e-4);
        assert(fabs(o_coeffs[18] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[19] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[20] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[21] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[22] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[23] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[24] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[25] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[26] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[27] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[28] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[29] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[30] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[31] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[32] - fast_logistic(2)) < 1e-4);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(fabs(((double*)coding_tmps)[9+0] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+1] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+2] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+3] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+4] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+5] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+6] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+7] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+8] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+9] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+10] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+11] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+12] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+13] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+14] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+15] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+16] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+17] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+18] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+19] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+20] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+21] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+22] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+23] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+24] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+25] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+26] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+27] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+28] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+29] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+30] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+31] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+32] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+33] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+34] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+35] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+36] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+37] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+38] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+39] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+40] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+41] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+42] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+43] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+44] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+45] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+46] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+47] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+48] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+49] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+50] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+51] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+52] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+53] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+54] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+55] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+56] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+57] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+58] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+59] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+60] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+61] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+62] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+63] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+64] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+65] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+66] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+67] - fast_logistic(1)) < 1e-4);
       	assert(fabs(((double*)coding_tmps)[9+68] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+69] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+70] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+71] - fast_logistic(1)) < 1e-4);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Logistic nonlinearity, NoSign polarity split and MaxKeepSign reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LOGISTIC;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NO_SIGN;
        enum reduce_type          reduce_type = MAX_KEEP_SIGN;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(fabs(o_coeffs[0] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[1] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[2] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[3] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[4] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[5] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[6] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[7] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[8] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[9] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[10] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[11] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[12] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[13] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[14] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[15] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[16] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[17] - fast_logistic(3)) < 1e-4);
        assert(fabs(o_coeffs[18] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[19] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[20] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[21] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[22] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[23] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[24] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[25] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[26] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[27] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[28] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[29] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[30] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[31] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[32] - fast_logistic(2)) < 1e-4);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(fabs(((double*)coding_tmps)[9+0] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+1] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+2] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+3] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+4] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+5] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+6] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+7] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+8] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+9] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+10] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+11] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+12] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+13] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+14] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+15] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+16] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+17] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+18] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+19] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+20] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+21] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+22] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+23] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+24] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+25] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+26] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+27] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+28] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+29] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+30] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+31] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+32] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+33] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+34] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+35] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+36] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+37] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+38] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+39] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+40] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+41] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+42] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+43] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+44] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+45] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+46] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+47] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+48] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+49] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+50] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+51] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+52] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+53] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+54] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+55] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+56] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+57] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+58] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+59] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+60] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+61] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+62] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+63] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+64] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+65] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+66] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+67] - fast_logistic(1)) < 1e-4);
       	assert(fabs(((double*)coding_tmps)[9+68] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+69] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+70] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+71] - fast_logistic(1)) < 1e-4);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Logistic nonlinearity, NoSign polarity split and SumAbs reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LOGISTIC;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NO_SIGN;
        enum reduce_type          reduce_type = SUM_ABS;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(fabs(o_coeffs[0] - (fast_logistic(1) + fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[1] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[2] - (fast_logistic(2) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[3] - (fast_logistic(1) + fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[4] - (fast_logistic(2) + fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[5] - (fast_logistic(2) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[6] - (fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[7] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[8] - (fast_logistic(2) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[9] - (fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[10] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[11] - (fast_logistic(2) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[12] - (fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[13] - (fast_logistic(2) + fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[14] - (fast_logistic(2) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[15] - (fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[16] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[17] - (fast_logistic(2) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[18] - (fast_logistic(1) + fast_logistic(1))) < 1e-4);
	assert(fabs(o_coeffs[19] - (fast_logistic(1) + fast_logistic(1) + fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[20] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[21] - (fast_logistic(1) + fast_logistic(1) + fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[22] - (fast_logistic(1) + fast_logistic(1))) < 1e-4);
	assert(fabs(o_coeffs[23] - (fast_logistic(1) + fast_logistic(1) + fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[24] - (fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[25] - (fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[26] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[27] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[28] - (fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[29] - (fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[30] - (fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[31] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[32] - (fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(fabs(((double*)coding_tmps)[9+0] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+1] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+2] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+3] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+4] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+5] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+6] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+7] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+8] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+9] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+10] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+11] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+12] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+13] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+14] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+15] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+16] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+17] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+18] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+19] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+20] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+21] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+22] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+23] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+24] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+25] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+26] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+27] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+28] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+29] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+30] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+31] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+32] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+33] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+34] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+35] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+36] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+37] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+38] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+39] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+40] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+41] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+42] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+43] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+44] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+45] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+46] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+47] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+48] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+49] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+50] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+51] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+52] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+53] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+54] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+55] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+56] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+57] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+58] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+59] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+60] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+61] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+62] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+63] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+64] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+65] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+66] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+67] - fast_logistic(1)) < 1e-4);
       	assert(fabs(((double*)coding_tmps)[9+68] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+69] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+70] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+71] - fast_logistic(1)) < 1e-4);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Logistic nonlinearity, NoSign polarity split and SumSqr reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LOGISTIC;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NO_SIGN;
        enum reduce_type          reduce_type = SUM_SQR;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(fabs(o_coeffs[0] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[1] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[2] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[3] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[4] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[5] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[6] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[7] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[8] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[9] - (fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[10] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[11] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[12] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[13] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[14] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[15] - (fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[16] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[17] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[18] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1))) < 1e-4);
	assert(fabs(o_coeffs[19] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[20] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[21] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[22] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1))) < 1e-4);
	assert(fabs(o_coeffs[23] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[24] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[25] - (fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[26] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[27] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[28] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[29] - (fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[30] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[31] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[32] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(fabs(((double*)coding_tmps)[9+0] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+1] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+2] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+3] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+4] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+5] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+6] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+7] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+8] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+9] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+10] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+11] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+12] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+13] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+14] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+15] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+16] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+17] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+18] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+19] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+20] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+21] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+22] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+23] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+24] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+25] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+26] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+27] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+28] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+29] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+30] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+31] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+32] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+33] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+34] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+35] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+36] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+37] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+38] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+39] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+40] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+41] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+42] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+43] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+44] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+45] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+46] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+47] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+48] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+49] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+50] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+51] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+52] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+53] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+54] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+55] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+56] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+57] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+58] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+59] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+60] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+61] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+62] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+63] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+64] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+65] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+66] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+67] - fast_logistic(1)) < 1e-4);
       	assert(fabs(((double*)coding_tmps)[9+68] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+69] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+70] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+71] - fast_logistic(1)) < 1e-4);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Logistic nonlinearity, KeepSign polarity split and Subsample reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LOGISTIC;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = KEEP_SIGN;
        enum reduce_type          reduce_type = SUBSAMPLE;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 18);
	assert(fabs(o_coeffs[0] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[1] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[2] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[3] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[4] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[5] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[6] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[7] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[8] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[9] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[10] - fast_logistic(-1)) < 1e-4);
	assert(fabs(o_coeffs[11] - fast_logistic(-1)) < 1e-4);
	assert(fabs(o_coeffs[12] - fast_logistic(-3)) < 1e-4);
	assert(fabs(o_coeffs[13] - fast_logistic(-3)) < 1e-4);
	assert(fabs(o_coeffs[14] - fast_logistic(-2)) < 1e-4);
	assert(fabs(o_coeffs[15] - fast_logistic(-1)) < 1e-4);
	assert(fabs(o_coeffs[16] - fast_logistic(-3)) < 1e-4);
	assert(fabs(o_coeffs[17] - fast_logistic(-2)) < 1e-4);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 16);
	assert(o_coeffs_idx[6] == 17);
	assert(o_coeffs_idx[7] == 18);
	assert(o_coeffs_idx[8] == 19);
	assert(o_coeffs_idx[9] == 20);
	assert(o_coeffs_idx[10] == 54+28);
	assert(o_coeffs_idx[11] == 54+29);
	assert(o_coeffs_idx[12] == 54+31);
	assert(o_coeffs_idx[13] == 54+32);
	assert(o_coeffs_idx[14] == 54+35);
	assert(o_coeffs_idx[15] == 54+45);
	assert(o_coeffs_idx[16] == 54+48);
	assert(o_coeffs_idx[17] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(fabs(((double*)coding_tmps)[9+0] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+1] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+2] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+3] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+4] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+5] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+6] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+7] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+8] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+9] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+10] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+11] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+12] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+13] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+14] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+15] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+16] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+17] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+18] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+19] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+20] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+21] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+22] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+23] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+24] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+25] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+26] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+27] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+28] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+29] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+30] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+31] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+32] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+33] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+34] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+35] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+36] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+37] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+38] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+39] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+40] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+41] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+42] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+43] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+44] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+45] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+46] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+47] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+48] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+49] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+50] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+51] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+52] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+53] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+54] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+55] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+56] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+57] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+58] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+59] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+60] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+61] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+62] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+63] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+64] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+65] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+66] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+67] - fast_logistic(-1)) < 1e-4);
       	assert(fabs(((double*)coding_tmps)[9+68] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+69] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+70] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+71] - fast_logistic(-1)) < 1e-4);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 0);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Logistic nonlinearity, KeepSign polarity split and MaxNoSign reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LOGISTIC;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = KEEP_SIGN;
        enum reduce_type          reduce_type = MAX_NO_SIGN;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(fabs(o_coeffs[0] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[1] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[2] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[3] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[4] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[5] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[6] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[7] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[8] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[9] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[10] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[11] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[12] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[13] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[14] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[15] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[16] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[17] - fast_logistic(3)) < 1e-4);
        assert(fabs(o_coeffs[18] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[19] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[20] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[21] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[22] - fast_logistic(1)) < 1e-4);
	assert(fabs(o_coeffs[23] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[24] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[25] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[26] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[27] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[28] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[29] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[30] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[31] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[32] - fast_logistic(2)) < 1e-4);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(fabs(((double*)coding_tmps)[9+0] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+1] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+2] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+3] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+4] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+5] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+6] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+7] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+8] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+9] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+10] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+11] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+12] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+13] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+14] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+15] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+16] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+17] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+18] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+19] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+20] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+21] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+22] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+23] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+24] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+25] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+26] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+27] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+28] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+29] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+30] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+31] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+32] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+33] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+34] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+35] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+36] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+37] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+38] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+39] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+40] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+41] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+42] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+43] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+44] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+45] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+46] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+47] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+48] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+49] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+50] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+51] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+52] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+53] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+54] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+55] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+56] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+57] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+58] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+59] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+60] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+61] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+62] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+63] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+64] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+65] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+66] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+67] - fast_logistic(-1)) < 1e-4);
       	assert(fabs(((double*)coding_tmps)[9+68] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+69] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+70] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+71] - fast_logistic(-1)) < 1e-4);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Logistic nonlinearity, KeepSign polarity split and MaxKeepSign reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LOGISTIC;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = KEEP_SIGN;
        enum reduce_type          reduce_type = MAX_KEEP_SIGN;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(fabs(o_coeffs[0] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[1] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[2] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[3] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[4] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[5] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[6] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[7] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[8] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[9] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[10] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[11] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[12] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[13] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[14] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[15] - fast_logistic(2)) < 1e-4);
	assert(fabs(o_coeffs[16] - fast_logistic(3)) < 1e-4);
	assert(fabs(o_coeffs[17] - fast_logistic(3)) < 1e-4);
        assert(fabs(o_coeffs[18] - fast_logistic(-1)) < 1e-4);
	assert(fabs(o_coeffs[19] - fast_logistic(-2)) < 1e-4);
	assert(fabs(o_coeffs[20] - fast_logistic(-3)) < 1e-4);
	assert(fabs(o_coeffs[21] - fast_logistic(-3)) < 1e-4);
	assert(fabs(o_coeffs[22] - fast_logistic(-1)) < 1e-4);
	assert(fabs(o_coeffs[23] - fast_logistic(-2)) < 1e-4);
	assert(fabs(o_coeffs[24] - fast_logistic(-2)) < 1e-4);
	assert(fabs(o_coeffs[25] - fast_logistic(-2)) < 1e-4);
	assert(fabs(o_coeffs[26] - fast_logistic(-3)) < 1e-4);
	assert(fabs(o_coeffs[27] - fast_logistic(-3)) < 1e-4);
	assert(fabs(o_coeffs[28] - fast_logistic(-2)) < 1e-4);
	assert(fabs(o_coeffs[29] - fast_logistic(-2)) < 1e-4);
	assert(fabs(o_coeffs[30] - fast_logistic(-2)) < 1e-4);
	assert(fabs(o_coeffs[31] - fast_logistic(-3)) < 1e-4);
	assert(fabs(o_coeffs[32] - fast_logistic(-2)) < 1e-4);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(fabs(((double*)coding_tmps)[9+0] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+1] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+2] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+3] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+4] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+5] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+6] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+7] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+8] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+9] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+10] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+11] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+12] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+13] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+14] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+15] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+16] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+17] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+18] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+19] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+20] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+21] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+22] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+23] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+24] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+25] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+26] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+27] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+28] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+29] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+30] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+31] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+32] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+33] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+34] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+35] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+36] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+37] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+38] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+39] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+40] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+41] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+42] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+43] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+44] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+45] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+46] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+47] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+48] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+49] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+50] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+51] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+52] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+53] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+54] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+55] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+56] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+57] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+58] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+59] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+60] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+61] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+62] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+63] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+64] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+65] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+66] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+67] - fast_logistic(-1)) < 1e-4);
       	assert(fabs(((double*)coding_tmps)[9+68] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+69] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+70] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+71] - fast_logistic(-1)) < 1e-4);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Logistic nonlinearity, KeepSign polarity split and SumAbs reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LOGISTIC;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = KEEP_SIGN;
        enum reduce_type          reduce_type = SUM_ABS;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(fabs(o_coeffs[0] - (fast_logistic(1) + fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[1] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[2] - (fast_logistic(2) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[3] - (fast_logistic(1) + fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[4] - (fast_logistic(2) + fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[5] - (fast_logistic(2) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[6] - (fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[7] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[8] - (fast_logistic(2) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[9] - (fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[10] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[11] - (fast_logistic(2) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[12] - (fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[13] - (fast_logistic(2) + fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[14] - (fast_logistic(2) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[15] - (fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[16] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[17] - (fast_logistic(2) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[18] - (fast_logistic(1) + fast_logistic(1))) < 1e-4);
	assert(fabs(o_coeffs[19] - (fast_logistic(1) + fast_logistic(1) + fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[20] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[21] - (fast_logistic(1) + fast_logistic(1) + fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[22] - (fast_logistic(1) + fast_logistic(1))) < 1e-4);
	assert(fabs(o_coeffs[23] - (fast_logistic(1) + fast_logistic(1) + fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[24] - (fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[25] - (fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[26] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[27] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[28] - (fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[29] - (fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[30] - (fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[31] - (fast_logistic(3) + fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[32] - (fast_logistic(1) + fast_logistic(2))) < 1e-4);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(fabs(((double*)coding_tmps)[9+0] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+1] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+2] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+3] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+4] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+5] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+6] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+7] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+8] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+9] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+10] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+11] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+12] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+13] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+14] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+15] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+16] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+17] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+18] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+19] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+20] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+21] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+22] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+23] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+24] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+25] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+26] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+27] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+28] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+29] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+30] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+31] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+32] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+33] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+34] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+35] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+36] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+37] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+38] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+39] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+40] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+41] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+42] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+43] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+44] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+45] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+46] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+47] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+48] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+49] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+50] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+51] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+52] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+53] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+54] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+55] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+56] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+57] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+58] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+59] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+60] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+61] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+62] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+63] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+64] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+65] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+66] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+67] - fast_logistic(-1)) < 1e-4);
       	assert(fabs(((double*)coding_tmps)[9+68] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+69] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+70] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+71] - fast_logistic(-1)) < 1e-4);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With Logistic nonlinearity, KeepSign polarity split and SumSqr reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = LOGISTIC;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = KEEP_SIGN;
        enum reduce_type          reduce_type = SUM_SQR;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(fabs(o_coeffs[0] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[1] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[2] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[3] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[4] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[5] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[6] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[7] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[8] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[9] - (fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[10] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[11] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[12] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[13] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[14] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[15] - (fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[16] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[17] - (fast_logistic(2)*fast_logistic(2) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[18] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1))) < 1e-4);
	assert(fabs(o_coeffs[19] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[20] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[21] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[22] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1))) < 1e-4);
	assert(fabs(o_coeffs[23] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[24] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[25] - (fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[26] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[27] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[28] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[29] - (fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[30] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(fabs(o_coeffs[31] - (fast_logistic(3)*fast_logistic(3) + fast_logistic(3)*fast_logistic(3))) < 1e-4);
	assert(fabs(o_coeffs[32] - (fast_logistic(1)*fast_logistic(1) + fast_logistic(2)*fast_logistic(2))) < 1e-4);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(fabs(((double*)coding_tmps)[9+0] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+1] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+2] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+3] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+4] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+5] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+6] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+7] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+8] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+9] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+10] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+11] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+12] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+13] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+14] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+15] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+16] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+17] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+18] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+19] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+20] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+21] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+22] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+23] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+24] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+25] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+26] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+27] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+28] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+29] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+30] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+31] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+32] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+33] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+34] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+35] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+36] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+37] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+38] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+39] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+40] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+41] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+42] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+43] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+44] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+45] - fast_logistic(-3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+46] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+47] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+48] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+49] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+50] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+51] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+52] - fast_logistic(1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+53] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+54] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+55] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+56] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+57] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+58] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+59] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+60] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+61] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+62] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+63] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+64] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+65] - fast_logistic(-2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+66] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+67] - fast_logistic(-1)) < 1e-4);
       	assert(fabs(((double*)coding_tmps)[9+68] - fast_logistic(3)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+69] - fast_logistic(-1)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+70] - fast_logistic(2)) < 1e-4);
	assert(fabs(((double*)coding_tmps)[9+71] - fast_logistic(-1)) < 1e-4);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With GlobalOrder nonlinearity, None polarity split and Subsample reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = GLOBAL_ORDER;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NONE;
        enum reduce_type          reduce_type = SUBSAMPLE;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 18);
	assert(o_coeffs[0] == 0.5);
	assert(o_coeffs[1] == 1);
	assert(o_coeffs[2] == 1);
	assert(o_coeffs[3] == 0.5);
	assert(o_coeffs[4] == 0.5);
	assert(o_coeffs[5] == 1);
	assert(o_coeffs[6] == 1);
	assert(o_coeffs[7] == 1);
	assert(o_coeffs[8] == 1);
	assert(o_coeffs[9] == 1);
	assert(o_coeffs[10] == -0.5);
	assert(o_coeffs[11] == -0.5);
	assert(o_coeffs[12] == -0.5);
	assert(o_coeffs[13] == -0.5);
	assert(o_coeffs[14] == -0.5);
	assert(o_coeffs[15] == -0.5);
	assert(o_coeffs[16] == -1);
	assert(o_coeffs[17] == -1);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 16);
	assert(o_coeffs_idx[6] == 17);
	assert(o_coeffs_idx[7] == 18);
	assert(o_coeffs_idx[8] == 19);
	assert(o_coeffs_idx[9] == 20);
	assert(o_coeffs_idx[10] == 28);
	assert(o_coeffs_idx[11] == 29);
	assert(o_coeffs_idx[12] == 31);
	assert(o_coeffs_idx[13] == 32);
	assert(o_coeffs_idx[14] == 35);
	assert(o_coeffs_idx[15] == 45);
	assert(o_coeffs_idx[16] == 48);
	assert(o_coeffs_idx[17] == 51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -0.5);
	assert(((double*)coding_tmps)[9+2] == 1);
	assert(((double*)coding_tmps)[9+3] == -0.5);
	assert(((double*)coding_tmps)[9+4] == 0.5);
	assert(((double*)coding_tmps)[9+5] == -1);
	assert(((double*)coding_tmps)[9+6] == 1);
	assert(((double*)coding_tmps)[9+7] == -0.5);
	assert(((double*)coding_tmps)[9+8] == 1);
	assert(((double*)coding_tmps)[9+9] == -0.5);
	assert(((double*)coding_tmps)[9+10] == 1);
	assert(((double*)coding_tmps)[9+11] == -0.5);
	assert(((double*)coding_tmps)[9+12] == 1);
	assert(((double*)coding_tmps)[9+13] == 0.5);
	assert(((double*)coding_tmps)[9+14] == 1);
	assert(((double*)coding_tmps)[9+15] == -0.5);
	assert(((double*)coding_tmps)[9+16] == 1);
	assert(((double*)coding_tmps)[9+17] == -0.5);
	assert(((double*)coding_tmps)[9+18] == 1);
	assert(((double*)coding_tmps)[9+19] == -0.5);
	assert(((double*)coding_tmps)[9+20] == 1);
	assert(((double*)coding_tmps)[9+21] == -0.5);
	assert(((double*)coding_tmps)[9+22] == 1);
	assert(((double*)coding_tmps)[9+23] == -0.5);
	assert(((double*)coding_tmps)[9+24] == 0.5);
	assert(((double*)coding_tmps)[9+25] == -1);
	assert(((double*)coding_tmps)[9+26] == 0.5);
	assert(((double*)coding_tmps)[9+27] == -1);
	assert(((double*)coding_tmps)[9+28] == 0.5);
	assert(((double*)coding_tmps)[9+29] == -1);
	assert(((double*)coding_tmps)[9+30] == 0.5);
	assert(((double*)coding_tmps)[9+31] == -1);
	assert(((double*)coding_tmps)[9+32] == 1);
	assert(((double*)coding_tmps)[9+33] == -0.5);
	assert(((double*)coding_tmps)[9+34] == 1);
	assert(((double*)coding_tmps)[9+35] == -0.5);
	assert(((double*)coding_tmps)[9+36] == 1);
	assert(((double*)coding_tmps)[9+37] == -0.5);
	assert(((double*)coding_tmps)[9+38] == 1);
	assert(((double*)coding_tmps)[9+39] == -0.5);
	assert(((double*)coding_tmps)[9+40] == 1);
	assert(((double*)coding_tmps)[9+41] == -0.5);
	assert(((double*)coding_tmps)[9+42] == 1);
	assert(((double*)coding_tmps)[9+43] == -0.5);
	assert(((double*)coding_tmps)[9+44] == 1);
	assert(((double*)coding_tmps)[9+45] == -0.5);
	assert(((double*)coding_tmps)[9+46] == 1);
	assert(((double*)coding_tmps)[9+47] == -0.5);
	assert(((double*)coding_tmps)[9+48] == 0.5);
	assert(((double*)coding_tmps)[9+49] == -1);
	assert(((double*)coding_tmps)[9+50] == 1);
	assert(((double*)coding_tmps)[9+51] == -0.5);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -0.5);
	assert(((double*)coding_tmps)[9+54] == 1);
	assert(((double*)coding_tmps)[9+55] == -0.5);
	assert(((double*)coding_tmps)[9+56] == 0.5);
	assert(((double*)coding_tmps)[9+57] == 1);
	assert(((double*)coding_tmps)[9+58] == 1);
	assert(((double*)coding_tmps)[9+59] == -0.5);
	assert(((double*)coding_tmps)[9+60] == 1);
	assert(((double*)coding_tmps)[9+61] == -0.5);
	assert(((double*)coding_tmps)[9+62] == 1);
	assert(((double*)coding_tmps)[9+63] == -0.5);
	assert(((double*)coding_tmps)[9+64] == 1);
	assert(((double*)coding_tmps)[9+65] == -0.5);
	assert(((double*)coding_tmps)[9+66] == 1);
	assert(((double*)coding_tmps)[9+67] == -0.5);
       	assert(((double*)coding_tmps)[9+68] == 1);
	assert(((double*)coding_tmps)[9+69] == -0.5);
	assert(((double*)coding_tmps)[9+70] == 1);
	assert(((double*)coding_tmps)[9+71] == -0.5);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 3);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 3);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 0);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With GlobalOrder nonlinearity, None polarity split and MaxNoSign reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = GLOBAL_ORDER;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NONE;
        enum reduce_type          reduce_type = MAX_NO_SIGN;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 0.5);
	assert(o_coeffs[1] == 1);
	assert(o_coeffs[2] == 1);
	assert(o_coeffs[3] == 1);
	assert(o_coeffs[4] == 1);
	assert(o_coeffs[5] == 1);
	assert(o_coeffs[6] == 1);
	assert(o_coeffs[7] == 1);
	assert(o_coeffs[8] == 1);
	assert(o_coeffs[9] == 1);
	assert(o_coeffs[10] == 1);
	assert(o_coeffs[11] == 1);
	assert(o_coeffs[12] == 1);
	assert(o_coeffs[13] == 1);
	assert(o_coeffs[14] == 1);
	assert(o_coeffs[15] == 0.5);
	assert(o_coeffs[16] == 1);
	assert(o_coeffs[17] == 1);
	assert(o_coeffs[18] == 0.5);
	assert(o_coeffs[19] == 0.5);
	assert(o_coeffs[20] == 0.5);
	assert(o_coeffs[21] == 0.5);
	assert(o_coeffs[22] == 0.5);
	assert(o_coeffs[23] == 0.5);
	assert(o_coeffs[24] == 0.5);
	assert(o_coeffs[25] == 0.5);
	assert(o_coeffs[26] == 1);
	assert(o_coeffs[27] == 0.5);
	assert(o_coeffs[28] == 0.5);
	assert(o_coeffs[29] == 0.5);
	assert(o_coeffs[30] == 1);
	assert(o_coeffs[31] == 1);
	assert(o_coeffs[32] == 1);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 28);
	assert(o_coeffs_idx[19] == 29);
	assert(o_coeffs_idx[20] == 31);
	assert(o_coeffs_idx[21] == 32);
	assert(o_coeffs_idx[22] == 34);
	assert(o_coeffs_idx[23] == 35);
	assert(o_coeffs_idx[24] == 36);
	assert(o_coeffs_idx[25] == 37);
	assert(o_coeffs_idx[26] == 39);
	assert(o_coeffs_idx[27] == 40);
	assert(o_coeffs_idx[28] == 42);
	assert(o_coeffs_idx[29] == 43);
	assert(o_coeffs_idx[30] == 45);
	assert(o_coeffs_idx[31] == 48);
	assert(o_coeffs_idx[32] == 51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -0.5);
	assert(((double*)coding_tmps)[9+2] == 1);
	assert(((double*)coding_tmps)[9+3] == -0.5);
	assert(((double*)coding_tmps)[9+4] == 0.5);
	assert(((double*)coding_tmps)[9+5] == -1);
	assert(((double*)coding_tmps)[9+6] == 1);
	assert(((double*)coding_tmps)[9+7] == -0.5);
	assert(((double*)coding_tmps)[9+8] == 1);
	assert(((double*)coding_tmps)[9+9] == -0.5);
	assert(((double*)coding_tmps)[9+10] == 1);
	assert(((double*)coding_tmps)[9+11] == -0.5);
	assert(((double*)coding_tmps)[9+12] == 1);
	assert(((double*)coding_tmps)[9+13] == 0.5);
	assert(((double*)coding_tmps)[9+14] == 1);
	assert(((double*)coding_tmps)[9+15] == -0.5);
	assert(((double*)coding_tmps)[9+16] == 1);
	assert(((double*)coding_tmps)[9+17] == -0.5);
	assert(((double*)coding_tmps)[9+18] == 1);
	assert(((double*)coding_tmps)[9+19] == -0.5);
	assert(((double*)coding_tmps)[9+20] == 1);
	assert(((double*)coding_tmps)[9+21] == -0.5);
	assert(((double*)coding_tmps)[9+22] == 1);
	assert(((double*)coding_tmps)[9+23] == -0.5);
	assert(((double*)coding_tmps)[9+24] == 0.5);
	assert(((double*)coding_tmps)[9+25] == -1);
	assert(((double*)coding_tmps)[9+26] == 0.5);
	assert(((double*)coding_tmps)[9+27] == -1);
	assert(((double*)coding_tmps)[9+28] == 0.5);
	assert(((double*)coding_tmps)[9+29] == -1);
	assert(((double*)coding_tmps)[9+30] == 0.5);
	assert(((double*)coding_tmps)[9+31] == -1);
	assert(((double*)coding_tmps)[9+32] == 1);
	assert(((double*)coding_tmps)[9+33] == -0.5);
	assert(((double*)coding_tmps)[9+34] == 1);
	assert(((double*)coding_tmps)[9+35] == -0.5);
	assert(((double*)coding_tmps)[9+36] == 1);
	assert(((double*)coding_tmps)[9+37] == -0.5);
	assert(((double*)coding_tmps)[9+38] == 1);
	assert(((double*)coding_tmps)[9+39] == -0.5);
	assert(((double*)coding_tmps)[9+40] == 1);
	assert(((double*)coding_tmps)[9+41] == -0.5);
	assert(((double*)coding_tmps)[9+42] == 1);
	assert(((double*)coding_tmps)[9+43] == -0.5);
	assert(((double*)coding_tmps)[9+44] == 1);
	assert(((double*)coding_tmps)[9+45] == -0.5);
	assert(((double*)coding_tmps)[9+46] == 1);
	assert(((double*)coding_tmps)[9+47] == -0.5);
	assert(((double*)coding_tmps)[9+48] == 0.5);
	assert(((double*)coding_tmps)[9+49] == -1);
	assert(((double*)coding_tmps)[9+50] == 1);
	assert(((double*)coding_tmps)[9+51] == -0.5);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -0.5);
	assert(((double*)coding_tmps)[9+54] == 1);
	assert(((double*)coding_tmps)[9+55] == -0.5);
	assert(((double*)coding_tmps)[9+56] == 0.5);
	assert(((double*)coding_tmps)[9+57] == 1);
	assert(((double*)coding_tmps)[9+58] == 1);
	assert(((double*)coding_tmps)[9+59] == -0.5);
	assert(((double*)coding_tmps)[9+60] == 1);
	assert(((double*)coding_tmps)[9+61] == -0.5);
	assert(((double*)coding_tmps)[9+62] == 1);
	assert(((double*)coding_tmps)[9+63] == -0.5);
	assert(((double*)coding_tmps)[9+64] == 1);
	assert(((double*)coding_tmps)[9+65] == -0.5);
	assert(((double*)coding_tmps)[9+66] == 1);
	assert(((double*)coding_tmps)[9+67] == -0.5);
       	assert(((double*)coding_tmps)[9+68] == 1);
	assert(((double*)coding_tmps)[9+69] == -0.5);
	assert(((double*)coding_tmps)[9+70] == 1);
	assert(((double*)coding_tmps)[9+71] == -0.5);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 3);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 3);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With GlobalOrder nonlinearity, None polarity split and MaxKeepSign reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = GLOBAL_ORDER;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NONE;
        enum reduce_type          reduce_type = MAX_KEEP_SIGN;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 0.5);
	assert(o_coeffs[1] == 1);
	assert(o_coeffs[2] == 1);
	assert(o_coeffs[3] == 1);
	assert(o_coeffs[4] == 1);
	assert(o_coeffs[5] == 1);
	assert(o_coeffs[6] == 1);
	assert(o_coeffs[7] == 1);
	assert(o_coeffs[8] == 1);
	assert(o_coeffs[9] == 1);
	assert(o_coeffs[10] == 1);
	assert(o_coeffs[11] == 1);
	assert(o_coeffs[12] == 1);
	assert(o_coeffs[13] == 1);
	assert(o_coeffs[14] == 1);
	assert(o_coeffs[15] == 0.5);
	assert(o_coeffs[16] == 1);
	assert(o_coeffs[17] == 1);
	assert(o_coeffs[18] == -0.5);
	assert(o_coeffs[19] == -0.5);
	assert(o_coeffs[20] == -0.5);
	assert(o_coeffs[21] == -0.5);
	assert(o_coeffs[22] == -0.5);
	assert(o_coeffs[23] == -0.5);
	assert(o_coeffs[24] == -0.5);
	assert(o_coeffs[25] == -0.5);
	assert(o_coeffs[26] == -1);
	assert(o_coeffs[27] == -0.5);
	assert(o_coeffs[28] == -0.5);
	assert(o_coeffs[29] == -0.5);
	assert(o_coeffs[30] == -1);
	assert(o_coeffs[31] == -1);
	assert(o_coeffs[32] == -1);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 28);
	assert(o_coeffs_idx[19] == 29);
	assert(o_coeffs_idx[20] == 31);
	assert(o_coeffs_idx[21] == 32);
	assert(o_coeffs_idx[22] == 34);
	assert(o_coeffs_idx[23] == 35);
	assert(o_coeffs_idx[24] == 36);
	assert(o_coeffs_idx[25] == 37);
	assert(o_coeffs_idx[26] == 39);
	assert(o_coeffs_idx[27] == 40);
	assert(o_coeffs_idx[28] == 42);
	assert(o_coeffs_idx[29] == 43);
	assert(o_coeffs_idx[30] == 45);
	assert(o_coeffs_idx[31] == 48);
	assert(o_coeffs_idx[32] == 51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -0.5);
	assert(((double*)coding_tmps)[9+2] == 1);
	assert(((double*)coding_tmps)[9+3] == -0.5);
	assert(((double*)coding_tmps)[9+4] == 0.5);
	assert(((double*)coding_tmps)[9+5] == -1);
	assert(((double*)coding_tmps)[9+6] == 1);
	assert(((double*)coding_tmps)[9+7] == -0.5);
	assert(((double*)coding_tmps)[9+8] == 1);
	assert(((double*)coding_tmps)[9+9] == -0.5);
	assert(((double*)coding_tmps)[9+10] == 1);
	assert(((double*)coding_tmps)[9+11] == -0.5);
	assert(((double*)coding_tmps)[9+12] == 1);
	assert(((double*)coding_tmps)[9+13] == 0.5);
	assert(((double*)coding_tmps)[9+14] == 1);
	assert(((double*)coding_tmps)[9+15] == -0.5);
	assert(((double*)coding_tmps)[9+16] == 1);
	assert(((double*)coding_tmps)[9+17] == -0.5);
	assert(((double*)coding_tmps)[9+18] == 1);
	assert(((double*)coding_tmps)[9+19] == -0.5);
	assert(((double*)coding_tmps)[9+20] == 1);
	assert(((double*)coding_tmps)[9+21] == -0.5);
	assert(((double*)coding_tmps)[9+22] == 1);
	assert(((double*)coding_tmps)[9+23] == -0.5);
	assert(((double*)coding_tmps)[9+24] == 0.5);
	assert(((double*)coding_tmps)[9+25] == -1);
	assert(((double*)coding_tmps)[9+26] == 0.5);
	assert(((double*)coding_tmps)[9+27] == -1);
	assert(((double*)coding_tmps)[9+28] == 0.5);
	assert(((double*)coding_tmps)[9+29] == -1);
	assert(((double*)coding_tmps)[9+30] == 0.5);
	assert(((double*)coding_tmps)[9+31] == -1);
	assert(((double*)coding_tmps)[9+32] == 1);
	assert(((double*)coding_tmps)[9+33] == -0.5);
	assert(((double*)coding_tmps)[9+34] == 1);
	assert(((double*)coding_tmps)[9+35] == -0.5);
	assert(((double*)coding_tmps)[9+36] == 1);
	assert(((double*)coding_tmps)[9+37] == -0.5);
	assert(((double*)coding_tmps)[9+38] == 1);
	assert(((double*)coding_tmps)[9+39] == -0.5);
	assert(((double*)coding_tmps)[9+40] == 1);
	assert(((double*)coding_tmps)[9+41] == -0.5);
	assert(((double*)coding_tmps)[9+42] == 1);
	assert(((double*)coding_tmps)[9+43] == -0.5);
	assert(((double*)coding_tmps)[9+44] == 1);
	assert(((double*)coding_tmps)[9+45] == -0.5);
	assert(((double*)coding_tmps)[9+46] == 1);
	assert(((double*)coding_tmps)[9+47] == -0.5);
	assert(((double*)coding_tmps)[9+48] == 0.5);
	assert(((double*)coding_tmps)[9+49] == -1);
	assert(((double*)coding_tmps)[9+50] == 1);
	assert(((double*)coding_tmps)[9+51] == -0.5);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -0.5);
	assert(((double*)coding_tmps)[9+54] == 1);
	assert(((double*)coding_tmps)[9+55] == -0.5);
	assert(((double*)coding_tmps)[9+56] == 0.5);
	assert(((double*)coding_tmps)[9+57] == 1);
	assert(((double*)coding_tmps)[9+58] == 1);
	assert(((double*)coding_tmps)[9+59] == -0.5);
	assert(((double*)coding_tmps)[9+60] == 1);
	assert(((double*)coding_tmps)[9+61] == -0.5);
	assert(((double*)coding_tmps)[9+62] == 1);
	assert(((double*)coding_tmps)[9+63] == -0.5);
	assert(((double*)coding_tmps)[9+64] == 1);
	assert(((double*)coding_tmps)[9+65] == -0.5);
	assert(((double*)coding_tmps)[9+66] == 1);
	assert(((double*)coding_tmps)[9+67] == -0.5);
       	assert(((double*)coding_tmps)[9+68] == 1);
	assert(((double*)coding_tmps)[9+69] == -0.5);
	assert(((double*)coding_tmps)[9+70] == 1);
	assert(((double*)coding_tmps)[9+71] == -0.5);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 3);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 3);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With GlobalOrder nonlinearity, None polarity split and SumAbs reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = GLOBAL_ORDER;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NONE;
        enum reduce_type          reduce_type = SUM_ABS;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 0.5 + 0.5 + 0.5);
	assert(o_coeffs[1] == 1 + 1);
	assert(o_coeffs[2] == 1 + 1);
	assert(o_coeffs[3] == 0.5 + 1 + 1);
	assert(o_coeffs[4] == 0.5 + 1 + 1);
	assert(o_coeffs[5] == 1 + 1);
	assert(o_coeffs[6] == 0.5 + 1);
	assert(o_coeffs[7] == 1 + 1);
	assert(o_coeffs[8] == 1 + 1);
	assert(o_coeffs[9] == 1);
	assert(o_coeffs[10] == 1 + 1);
	assert(o_coeffs[11] == 1 + 1);
	assert(o_coeffs[12] == 1 + 1);
	assert(o_coeffs[13] == 0.5 + 1 + 1);
	assert(o_coeffs[14] == 1 + 1);
	assert(o_coeffs[15] == 0.5);
	assert(o_coeffs[16] == 1 + 1);
	assert(o_coeffs[17] == 1 + 1);
	assert(o_coeffs[18] == 0.5 + 0.5);
	assert(o_coeffs[19] == 0.5 + 0.5 + 0.5 + 0.5);
	assert(o_coeffs[20] == 0.5 + 0.5);
	assert(o_coeffs[21] == 0.5 + 0.5 + 0.5 + 0.5);
	assert(o_coeffs[22] == 0.5 + 0.5);
	assert(o_coeffs[23] == 0.5 + 0.5 + 0.5 + 0.5);
	assert(o_coeffs[24] == 0.5 + 0.5);
	assert(o_coeffs[25] == 0.5);
	assert(o_coeffs[26] == 1 + 1);
	assert(o_coeffs[27] == 0.5 + 0.5);
	assert(o_coeffs[28] == 0.5 + 0.5);
	assert(o_coeffs[29] == 0.5);
	assert(o_coeffs[30] == 0.5 + 1);
	assert(o_coeffs[31] == 1 + 1);
	assert(o_coeffs[32] == 0.5 + 1);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 28);
	assert(o_coeffs_idx[19] == 29);
	assert(o_coeffs_idx[20] == 31);
	assert(o_coeffs_idx[21] == 32);
	assert(o_coeffs_idx[22] == 34);
	assert(o_coeffs_idx[23] == 35);
	assert(o_coeffs_idx[24] == 36);
	assert(o_coeffs_idx[25] == 37);
	assert(o_coeffs_idx[26] == 39);
	assert(o_coeffs_idx[27] == 40);
	assert(o_coeffs_idx[28] == 42);
	assert(o_coeffs_idx[29] == 43);
	assert(o_coeffs_idx[30] == 45);
	assert(o_coeffs_idx[31] == 48);
	assert(o_coeffs_idx[32] == 51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -0.5);
	assert(((double*)coding_tmps)[9+2] == 1);
	assert(((double*)coding_tmps)[9+3] == -0.5);
	assert(((double*)coding_tmps)[9+4] == 0.5);
	assert(((double*)coding_tmps)[9+5] == -1);
	assert(((double*)coding_tmps)[9+6] == 1);
	assert(((double*)coding_tmps)[9+7] == -0.5);
	assert(((double*)coding_tmps)[9+8] == 1);
	assert(((double*)coding_tmps)[9+9] == -0.5);
	assert(((double*)coding_tmps)[9+10] == 1);
	assert(((double*)coding_tmps)[9+11] == -0.5);
	assert(((double*)coding_tmps)[9+12] == 1);
	assert(((double*)coding_tmps)[9+13] == 0.5);
	assert(((double*)coding_tmps)[9+14] == 1);
	assert(((double*)coding_tmps)[9+15] == -0.5);
	assert(((double*)coding_tmps)[9+16] == 1);
	assert(((double*)coding_tmps)[9+17] == -0.5);
	assert(((double*)coding_tmps)[9+18] == 1);
	assert(((double*)coding_tmps)[9+19] == -0.5);
	assert(((double*)coding_tmps)[9+20] == 1);
	assert(((double*)coding_tmps)[9+21] == -0.5);
	assert(((double*)coding_tmps)[9+22] == 1);
	assert(((double*)coding_tmps)[9+23] == -0.5);
	assert(((double*)coding_tmps)[9+24] == 0.5);
	assert(((double*)coding_tmps)[9+25] == -1);
	assert(((double*)coding_tmps)[9+26] == 0.5);
	assert(((double*)coding_tmps)[9+27] == -1);
	assert(((double*)coding_tmps)[9+28] == 0.5);
	assert(((double*)coding_tmps)[9+29] == -1);
	assert(((double*)coding_tmps)[9+30] == 0.5);
	assert(((double*)coding_tmps)[9+31] == -1);
	assert(((double*)coding_tmps)[9+32] == 1);
	assert(((double*)coding_tmps)[9+33] == -0.5);
	assert(((double*)coding_tmps)[9+34] == 1);
	assert(((double*)coding_tmps)[9+35] == -0.5);
	assert(((double*)coding_tmps)[9+36] == 1);
	assert(((double*)coding_tmps)[9+37] == -0.5);
	assert(((double*)coding_tmps)[9+38] == 1);
	assert(((double*)coding_tmps)[9+39] == -0.5);
	assert(((double*)coding_tmps)[9+40] == 1);
	assert(((double*)coding_tmps)[9+41] == -0.5);
	assert(((double*)coding_tmps)[9+42] == 1);
	assert(((double*)coding_tmps)[9+43] == -0.5);
	assert(((double*)coding_tmps)[9+44] == 1);
	assert(((double*)coding_tmps)[9+45] == -0.5);
	assert(((double*)coding_tmps)[9+46] == 1);
	assert(((double*)coding_tmps)[9+47] == -0.5);
	assert(((double*)coding_tmps)[9+48] == 0.5);
	assert(((double*)coding_tmps)[9+49] == -1);
	assert(((double*)coding_tmps)[9+50] == 1);
	assert(((double*)coding_tmps)[9+51] == -0.5);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -0.5);
	assert(((double*)coding_tmps)[9+54] == 1);
	assert(((double*)coding_tmps)[9+55] == -0.5);
	assert(((double*)coding_tmps)[9+56] == 0.5);
	assert(((double*)coding_tmps)[9+57] == 1);
	assert(((double*)coding_tmps)[9+58] == 1);
	assert(((double*)coding_tmps)[9+59] == -0.5);
	assert(((double*)coding_tmps)[9+60] == 1);
	assert(((double*)coding_tmps)[9+61] == -0.5);
	assert(((double*)coding_tmps)[9+62] == 1);
	assert(((double*)coding_tmps)[9+63] == -0.5);
	assert(((double*)coding_tmps)[9+64] == 1);
	assert(((double*)coding_tmps)[9+65] == -0.5);
	assert(((double*)coding_tmps)[9+66] == 1);
	assert(((double*)coding_tmps)[9+67] == -0.5);
       	assert(((double*)coding_tmps)[9+68] == 1);
	assert(((double*)coding_tmps)[9+69] == -0.5);
	assert(((double*)coding_tmps)[9+70] == 1);
	assert(((double*)coding_tmps)[9+71] == -0.5);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 3);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 3);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With GlobalOrder nonlinearity, None polarity split and SumSqr reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = GLOBAL_ORDER;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NONE;
        enum reduce_type          reduce_type = SUM_SQR;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 0.5*0.5 + 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[1] == 1*1 + 1*1);
	assert(o_coeffs[2] == 1*1 + 1*1);
	assert(o_coeffs[3] == 0.5*0.5 + 1*1 + 1*1);
	assert(o_coeffs[4] == 0.5*0.5 + 1*1 + 1*1);
	assert(o_coeffs[5] == 1*1 + 1*1);
	assert(o_coeffs[6] == 0.5*0.5 + 1*1);
	assert(o_coeffs[7] == 1*1 + 1*1);
	assert(o_coeffs[8] == 1*1 + 1*1);
	assert(o_coeffs[9] == 1*1);
	assert(o_coeffs[10] == 1*1 + 1*1);
	assert(o_coeffs[11] == 1*1 + 1*1);
	assert(o_coeffs[12] == 1*1 + 1*1);
	assert(o_coeffs[13] == 0.5*0.5 + 1*1 + 1*1);
	assert(o_coeffs[14] == 1*1 + 1*1);
	assert(o_coeffs[15] == 0.5*0.5);
	assert(o_coeffs[16] == 1*1 + 1*1);
	assert(o_coeffs[17] == 1*1 + 1*1);
	assert(o_coeffs[18] == 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[19] == 0.5*0.5 + 0.5*0.5 + 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[20] == 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[21] == 0.5*0.5 + 0.5*0.5 + 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[22] == 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[23] == 0.5*0.5 + 0.5*0.5 + 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[24] == 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[25] == 0.5*0.5);
	assert(o_coeffs[26] == 1*1 + 1*1);
	assert(o_coeffs[27] == 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[28] == 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[29] == 0.5*0.5);
	assert(o_coeffs[30] == 0.5*0.5 + 1*1);
	assert(o_coeffs[31] == 1*1 + 1*1);
	assert(o_coeffs[32] == 0.5*0.5 + 1*1);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 28);
	assert(o_coeffs_idx[19] == 29);
	assert(o_coeffs_idx[20] == 31);
	assert(o_coeffs_idx[21] == 32);
	assert(o_coeffs_idx[22] == 34);
	assert(o_coeffs_idx[23] == 35);
	assert(o_coeffs_idx[24] == 36);
	assert(o_coeffs_idx[25] == 37);
	assert(o_coeffs_idx[26] == 39);
	assert(o_coeffs_idx[27] == 40);
	assert(o_coeffs_idx[28] == 42);
	assert(o_coeffs_idx[29] == 43);
	assert(o_coeffs_idx[30] == 45);
	assert(o_coeffs_idx[31] == 48);
	assert(o_coeffs_idx[32] == 51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -0.5);
	assert(((double*)coding_tmps)[9+2] == 1);
	assert(((double*)coding_tmps)[9+3] == -0.5);
	assert(((double*)coding_tmps)[9+4] == 0.5);
	assert(((double*)coding_tmps)[9+5] == -1);
	assert(((double*)coding_tmps)[9+6] == 1);
	assert(((double*)coding_tmps)[9+7] == -0.5);
	assert(((double*)coding_tmps)[9+8] == 1);
	assert(((double*)coding_tmps)[9+9] == -0.5);
	assert(((double*)coding_tmps)[9+10] == 1);
	assert(((double*)coding_tmps)[9+11] == -0.5);
	assert(((double*)coding_tmps)[9+12] == 1);
	assert(((double*)coding_tmps)[9+13] == 0.5);
	assert(((double*)coding_tmps)[9+14] == 1);
	assert(((double*)coding_tmps)[9+15] == -0.5);
	assert(((double*)coding_tmps)[9+16] == 1);
	assert(((double*)coding_tmps)[9+17] == -0.5);
	assert(((double*)coding_tmps)[9+18] == 1);
	assert(((double*)coding_tmps)[9+19] == -0.5);
	assert(((double*)coding_tmps)[9+20] == 1);
	assert(((double*)coding_tmps)[9+21] == -0.5);
	assert(((double*)coding_tmps)[9+22] == 1);
	assert(((double*)coding_tmps)[9+23] == -0.5);
	assert(((double*)coding_tmps)[9+24] == 0.5);
	assert(((double*)coding_tmps)[9+25] == -1);
	assert(((double*)coding_tmps)[9+26] == 0.5);
	assert(((double*)coding_tmps)[9+27] == -1);
	assert(((double*)coding_tmps)[9+28] == 0.5);
	assert(((double*)coding_tmps)[9+29] == -1);
	assert(((double*)coding_tmps)[9+30] == 0.5);
	assert(((double*)coding_tmps)[9+31] == -1);
	assert(((double*)coding_tmps)[9+32] == 1);
	assert(((double*)coding_tmps)[9+33] == -0.5);
	assert(((double*)coding_tmps)[9+34] == 1);
	assert(((double*)coding_tmps)[9+35] == -0.5);
	assert(((double*)coding_tmps)[9+36] == 1);
	assert(((double*)coding_tmps)[9+37] == -0.5);
	assert(((double*)coding_tmps)[9+38] == 1);
	assert(((double*)coding_tmps)[9+39] == -0.5);
	assert(((double*)coding_tmps)[9+40] == 1);
	assert(((double*)coding_tmps)[9+41] == -0.5);
	assert(((double*)coding_tmps)[9+42] == 1);
	assert(((double*)coding_tmps)[9+43] == -0.5);
	assert(((double*)coding_tmps)[9+44] == 1);
	assert(((double*)coding_tmps)[9+45] == -0.5);
	assert(((double*)coding_tmps)[9+46] == 1);
	assert(((double*)coding_tmps)[9+47] == -0.5);
	assert(((double*)coding_tmps)[9+48] == 0.5);
	assert(((double*)coding_tmps)[9+49] == -1);
	assert(((double*)coding_tmps)[9+50] == 1);
	assert(((double*)coding_tmps)[9+51] == -0.5);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -0.5);
	assert(((double*)coding_tmps)[9+54] == 1);
	assert(((double*)coding_tmps)[9+55] == -0.5);
	assert(((double*)coding_tmps)[9+56] == 0.5);
	assert(((double*)coding_tmps)[9+57] == 1);
	assert(((double*)coding_tmps)[9+58] == 1);
	assert(((double*)coding_tmps)[9+59] == -0.5);
	assert(((double*)coding_tmps)[9+60] == 1);
	assert(((double*)coding_tmps)[9+61] == -0.5);
	assert(((double*)coding_tmps)[9+62] == 1);
	assert(((double*)coding_tmps)[9+63] == -0.5);
	assert(((double*)coding_tmps)[9+64] == 1);
	assert(((double*)coding_tmps)[9+65] == -0.5);
	assert(((double*)coding_tmps)[9+66] == 1);
	assert(((double*)coding_tmps)[9+67] == -0.5);
       	assert(((double*)coding_tmps)[9+68] == 1);
	assert(((double*)coding_tmps)[9+69] == -0.5);
	assert(((double*)coding_tmps)[9+70] == 1);
	assert(((double*)coding_tmps)[9+71] == -0.5);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 5);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 4);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 3);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 3);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 3);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With GlobalOrder nonlinearity, NoSign polarity split and Subsample reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = GLOBAL_ORDER;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NO_SIGN;
        enum reduce_type          reduce_type = SUBSAMPLE;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 18);
	assert(o_coeffs[0] == 0.5);
	assert(o_coeffs[1] == 1);
	assert(o_coeffs[2] == 1);
	assert(o_coeffs[3] == 0.5);
	assert(o_coeffs[4] == 0.5);
	assert(o_coeffs[5] == 1);
	assert(o_coeffs[6] == 1);
	assert(o_coeffs[7] == 1);
	assert(o_coeffs[8] == 1);
	assert(o_coeffs[9] == 1);
	assert(o_coeffs[10] == 0.5);
	assert(o_coeffs[11] == 0.5);
	assert(o_coeffs[12] == 0.5);
	assert(o_coeffs[13] == 0.5);
	assert(o_coeffs[14] == 0.5);
	assert(o_coeffs[15] == 0.5);
	assert(o_coeffs[16] == 1);
	assert(o_coeffs[17] == 1);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 16);
	assert(o_coeffs_idx[6] == 17);
	assert(o_coeffs_idx[7] == 18);
	assert(o_coeffs_idx[8] == 19);
	assert(o_coeffs_idx[9] == 20);
	assert(o_coeffs_idx[10] == 54+28);
	assert(o_coeffs_idx[11] == 54+29);
	assert(o_coeffs_idx[12] == 54+31);
	assert(o_coeffs_idx[13] == 54+32);
	assert(o_coeffs_idx[14] == 54+35);
	assert(o_coeffs_idx[15] == 54+45);
	assert(o_coeffs_idx[16] == 54+48);
	assert(o_coeffs_idx[17] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == 0.5);
	assert(((double*)coding_tmps)[9+2] == 1);
	assert(((double*)coding_tmps)[9+3] == 0.5);
	assert(((double*)coding_tmps)[9+4] == 0.5);
	assert(((double*)coding_tmps)[9+5] == 1);
	assert(((double*)coding_tmps)[9+6] == 1);
	assert(((double*)coding_tmps)[9+7] == 0.5);
	assert(((double*)coding_tmps)[9+8] == 1);
	assert(((double*)coding_tmps)[9+9] == 0.5);
	assert(((double*)coding_tmps)[9+10] == 1);
	assert(((double*)coding_tmps)[9+11] == 0.5);
	assert(((double*)coding_tmps)[9+12] == 1);
	assert(((double*)coding_tmps)[9+13] == 0.5);
	assert(((double*)coding_tmps)[9+14] == 1);
	assert(((double*)coding_tmps)[9+15] == 0.5);
	assert(((double*)coding_tmps)[9+16] == 1);
	assert(((double*)coding_tmps)[9+17] == 0.5);
	assert(((double*)coding_tmps)[9+18] == 1);
	assert(((double*)coding_tmps)[9+19] == 0.5);
	assert(((double*)coding_tmps)[9+20] == 1);
	assert(((double*)coding_tmps)[9+21] == 0.5);
	assert(((double*)coding_tmps)[9+22] == 1);
	assert(((double*)coding_tmps)[9+23] == 0.5);
	assert(((double*)coding_tmps)[9+24] == 0.5);
	assert(((double*)coding_tmps)[9+25] == 1);
	assert(((double*)coding_tmps)[9+26] == 0.5);
	assert(((double*)coding_tmps)[9+27] == 1);
	assert(((double*)coding_tmps)[9+28] == 0.5);
	assert(((double*)coding_tmps)[9+29] == 1);
	assert(((double*)coding_tmps)[9+30] == 0.5);
	assert(((double*)coding_tmps)[9+31] == 1);
	assert(((double*)coding_tmps)[9+32] == 1);
	assert(((double*)coding_tmps)[9+33] == 0.5);
	assert(((double*)coding_tmps)[9+34] == 1);
	assert(((double*)coding_tmps)[9+35] == 0.5);
	assert(((double*)coding_tmps)[9+36] == 1);
	assert(((double*)coding_tmps)[9+37] == 0.5);
	assert(((double*)coding_tmps)[9+38] == 1);
	assert(((double*)coding_tmps)[9+39] == 0.5);
	assert(((double*)coding_tmps)[9+40] == 1);
	assert(((double*)coding_tmps)[9+41] == 0.5);
	assert(((double*)coding_tmps)[9+42] == 1);
	assert(((double*)coding_tmps)[9+43] == 0.5);
	assert(((double*)coding_tmps)[9+44] == 1);
	assert(((double*)coding_tmps)[9+45] == 0.5);
	assert(((double*)coding_tmps)[9+46] == 1);
	assert(((double*)coding_tmps)[9+47] == 0.5);
	assert(((double*)coding_tmps)[9+48] == 0.5);
	assert(((double*)coding_tmps)[9+49] == 1);
	assert(((double*)coding_tmps)[9+50] == 1);
	assert(((double*)coding_tmps)[9+51] == 0.5);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == 0.5);
	assert(((double*)coding_tmps)[9+54] == 1);
	assert(((double*)coding_tmps)[9+55] == 0.5);
	assert(((double*)coding_tmps)[9+56] == 0.5);
	assert(((double*)coding_tmps)[9+57] == 1);
	assert(((double*)coding_tmps)[9+58] == 1);
	assert(((double*)coding_tmps)[9+59] == 0.5);
	assert(((double*)coding_tmps)[9+60] == 1);
	assert(((double*)coding_tmps)[9+61] == 0.5);
	assert(((double*)coding_tmps)[9+62] == 1);
	assert(((double*)coding_tmps)[9+63] == 0.5);
	assert(((double*)coding_tmps)[9+64] == 1);
	assert(((double*)coding_tmps)[9+65] == 0.5);
	assert(((double*)coding_tmps)[9+66] == 1);
	assert(((double*)coding_tmps)[9+67] == 0.5);
       	assert(((double*)coding_tmps)[9+68] == 1);
	assert(((double*)coding_tmps)[9+69] == 0.5);
	assert(((double*)coding_tmps)[9+70] == 1);
	assert(((double*)coding_tmps)[9+71] == 0.5);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 0);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With GlobalOrder nonlinearity, NoSign polarity split and MaxNoSign reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = GLOBAL_ORDER;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NO_SIGN;
        enum reduce_type          reduce_type = MAX_NO_SIGN;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 0.5);
	assert(o_coeffs[1] == 1);
	assert(o_coeffs[2] == 1);
	assert(o_coeffs[3] == 1);
	assert(o_coeffs[4] == 1);
	assert(o_coeffs[5] == 1);
	assert(o_coeffs[6] == 1);
	assert(o_coeffs[7] == 1);
	assert(o_coeffs[8] == 1);
	assert(o_coeffs[9] == 1);
	assert(o_coeffs[10] == 1);
	assert(o_coeffs[11] == 1);
	assert(o_coeffs[12] == 1);
	assert(o_coeffs[13] == 1);
	assert(o_coeffs[14] == 1);
	assert(o_coeffs[15] == 0.5);
	assert(o_coeffs[16] == 1);
	assert(o_coeffs[17] == 1);
	assert(o_coeffs[18] == 0.5);
	assert(o_coeffs[19] == 0.5);
	assert(o_coeffs[20] == 0.5);
	assert(o_coeffs[21] == 0.5);
	assert(o_coeffs[22] == 0.5);
	assert(o_coeffs[23] == 0.5);
	assert(o_coeffs[24] == 0.5);
	assert(o_coeffs[25] == 0.5);
	assert(o_coeffs[26] == 1);
	assert(o_coeffs[27] == 0.5);
	assert(o_coeffs[28] == 0.5);
	assert(o_coeffs[29] == 0.5);
	assert(o_coeffs[30] == 1);
	assert(o_coeffs[31] == 1);
	assert(o_coeffs[32] == 1);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == 0.5);
	assert(((double*)coding_tmps)[9+2] == 1);
	assert(((double*)coding_tmps)[9+3] == 0.5);
	assert(((double*)coding_tmps)[9+4] == 0.5);
	assert(((double*)coding_tmps)[9+5] == 1);
	assert(((double*)coding_tmps)[9+6] == 1);
	assert(((double*)coding_tmps)[9+7] == 0.5);
	assert(((double*)coding_tmps)[9+8] == 1);
	assert(((double*)coding_tmps)[9+9] == 0.5);
	assert(((double*)coding_tmps)[9+10] == 1);
	assert(((double*)coding_tmps)[9+11] == 0.5);
	assert(((double*)coding_tmps)[9+12] == 1);
	assert(((double*)coding_tmps)[9+13] == 0.5);
	assert(((double*)coding_tmps)[9+14] == 1);
	assert(((double*)coding_tmps)[9+15] == 0.5);
	assert(((double*)coding_tmps)[9+16] == 1);
	assert(((double*)coding_tmps)[9+17] == 0.5);
	assert(((double*)coding_tmps)[9+18] == 1);
	assert(((double*)coding_tmps)[9+19] == 0.5);
	assert(((double*)coding_tmps)[9+20] == 1);
	assert(((double*)coding_tmps)[9+21] == 0.5);
	assert(((double*)coding_tmps)[9+22] == 1);
	assert(((double*)coding_tmps)[9+23] == 0.5);
	assert(((double*)coding_tmps)[9+24] == 0.5);
	assert(((double*)coding_tmps)[9+25] == 1);
	assert(((double*)coding_tmps)[9+26] == 0.5);
	assert(((double*)coding_tmps)[9+27] == 1);
	assert(((double*)coding_tmps)[9+28] == 0.5);
	assert(((double*)coding_tmps)[9+29] == 1);
	assert(((double*)coding_tmps)[9+30] == 0.5);
	assert(((double*)coding_tmps)[9+31] == 1);
	assert(((double*)coding_tmps)[9+32] == 1);
	assert(((double*)coding_tmps)[9+33] == 0.5);
	assert(((double*)coding_tmps)[9+34] == 1);
	assert(((double*)coding_tmps)[9+35] == 0.5);
	assert(((double*)coding_tmps)[9+36] == 1);
	assert(((double*)coding_tmps)[9+37] == 0.5);
	assert(((double*)coding_tmps)[9+38] == 1);
	assert(((double*)coding_tmps)[9+39] == 0.5);
	assert(((double*)coding_tmps)[9+40] == 1);
	assert(((double*)coding_tmps)[9+41] == 0.5);
	assert(((double*)coding_tmps)[9+42] == 1);
	assert(((double*)coding_tmps)[9+43] == 0.5);
	assert(((double*)coding_tmps)[9+44] == 1);
	assert(((double*)coding_tmps)[9+45] == 0.5);
	assert(((double*)coding_tmps)[9+46] == 1);
	assert(((double*)coding_tmps)[9+47] == 0.5);
	assert(((double*)coding_tmps)[9+48] == 0.5);
	assert(((double*)coding_tmps)[9+49] == 1);
	assert(((double*)coding_tmps)[9+50] == 1);
	assert(((double*)coding_tmps)[9+51] == 0.5);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == 0.5);
	assert(((double*)coding_tmps)[9+54] == 1);
	assert(((double*)coding_tmps)[9+55] == 0.5);
	assert(((double*)coding_tmps)[9+56] == 0.5);
	assert(((double*)coding_tmps)[9+57] == 1);
	assert(((double*)coding_tmps)[9+58] == 1);
	assert(((double*)coding_tmps)[9+59] == 0.5);
	assert(((double*)coding_tmps)[9+60] == 1);
	assert(((double*)coding_tmps)[9+61] == 0.5);
	assert(((double*)coding_tmps)[9+62] == 1);
	assert(((double*)coding_tmps)[9+63] == 0.5);
	assert(((double*)coding_tmps)[9+64] == 1);
	assert(((double*)coding_tmps)[9+65] == 0.5);
	assert(((double*)coding_tmps)[9+66] == 1);
	assert(((double*)coding_tmps)[9+67] == 0.5);
       	assert(((double*)coding_tmps)[9+68] == 1);
	assert(((double*)coding_tmps)[9+69] == 0.5);
	assert(((double*)coding_tmps)[9+70] == 1);
	assert(((double*)coding_tmps)[9+71] == 0.5);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With GlobalOrder nonlinearity, NoSign polarity split and MaxKeepSign reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = GLOBAL_ORDER;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NO_SIGN;
        enum reduce_type          reduce_type = MAX_KEEP_SIGN;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 0.5);
	assert(o_coeffs[1] == 1);
	assert(o_coeffs[2] == 1);
	assert(o_coeffs[3] == 1);
	assert(o_coeffs[4] == 1);
	assert(o_coeffs[5] == 1);
	assert(o_coeffs[6] == 1);
	assert(o_coeffs[7] == 1);
	assert(o_coeffs[8] == 1);
	assert(o_coeffs[9] == 1);
	assert(o_coeffs[10] == 1);
	assert(o_coeffs[11] == 1);
	assert(o_coeffs[12] == 1);
	assert(o_coeffs[13] == 1);
	assert(o_coeffs[14] == 1);
	assert(o_coeffs[15] == 0.5);
	assert(o_coeffs[16] == 1);
	assert(o_coeffs[17] == 1);
	assert(o_coeffs[18] == 0.5);
	assert(o_coeffs[19] == 0.5);
	assert(o_coeffs[20] == 0.5);
	assert(o_coeffs[21] == 0.5);
	assert(o_coeffs[22] == 0.5);
	assert(o_coeffs[23] == 0.5);
	assert(o_coeffs[24] == 0.5);
	assert(o_coeffs[25] == 0.5);
	assert(o_coeffs[26] == 1);
	assert(o_coeffs[27] == 0.5);
	assert(o_coeffs[28] == 0.5);
	assert(o_coeffs[29] == 0.5);
	assert(o_coeffs[30] == 1);
	assert(o_coeffs[31] == 1);
	assert(o_coeffs[32] == 1);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == 0.5);
	assert(((double*)coding_tmps)[9+2] == 1);
	assert(((double*)coding_tmps)[9+3] == 0.5);
	assert(((double*)coding_tmps)[9+4] == 0.5);
	assert(((double*)coding_tmps)[9+5] == 1);
	assert(((double*)coding_tmps)[9+6] == 1);
	assert(((double*)coding_tmps)[9+7] == 0.5);
	assert(((double*)coding_tmps)[9+8] == 1);
	assert(((double*)coding_tmps)[9+9] == 0.5);
	assert(((double*)coding_tmps)[9+10] == 1);
	assert(((double*)coding_tmps)[9+11] == 0.5);
	assert(((double*)coding_tmps)[9+12] == 1);
	assert(((double*)coding_tmps)[9+13] == 0.5);
	assert(((double*)coding_tmps)[9+14] == 1);
	assert(((double*)coding_tmps)[9+15] == 0.5);
	assert(((double*)coding_tmps)[9+16] == 1);
	assert(((double*)coding_tmps)[9+17] == 0.5);
	assert(((double*)coding_tmps)[9+18] == 1);
	assert(((double*)coding_tmps)[9+19] == 0.5);
	assert(((double*)coding_tmps)[9+20] == 1);
	assert(((double*)coding_tmps)[9+21] == 0.5);
	assert(((double*)coding_tmps)[9+22] == 1);
	assert(((double*)coding_tmps)[9+23] == 0.5);
	assert(((double*)coding_tmps)[9+24] == 0.5);
	assert(((double*)coding_tmps)[9+25] == 1);
	assert(((double*)coding_tmps)[9+26] == 0.5);
	assert(((double*)coding_tmps)[9+27] == 1);
	assert(((double*)coding_tmps)[9+28] == 0.5);
	assert(((double*)coding_tmps)[9+29] == 1);
	assert(((double*)coding_tmps)[9+30] == 0.5);
	assert(((double*)coding_tmps)[9+31] == 1);
	assert(((double*)coding_tmps)[9+32] == 1);
	assert(((double*)coding_tmps)[9+33] == 0.5);
	assert(((double*)coding_tmps)[9+34] == 1);
	assert(((double*)coding_tmps)[9+35] == 0.5);
	assert(((double*)coding_tmps)[9+36] == 1);
	assert(((double*)coding_tmps)[9+37] == 0.5);
	assert(((double*)coding_tmps)[9+38] == 1);
	assert(((double*)coding_tmps)[9+39] == 0.5);
	assert(((double*)coding_tmps)[9+40] == 1);
	assert(((double*)coding_tmps)[9+41] == 0.5);
	assert(((double*)coding_tmps)[9+42] == 1);
	assert(((double*)coding_tmps)[9+43] == 0.5);
	assert(((double*)coding_tmps)[9+44] == 1);
	assert(((double*)coding_tmps)[9+45] == 0.5);
	assert(((double*)coding_tmps)[9+46] == 1);
	assert(((double*)coding_tmps)[9+47] == 0.5);
	assert(((double*)coding_tmps)[9+48] == 0.5);
	assert(((double*)coding_tmps)[9+49] == 1);
	assert(((double*)coding_tmps)[9+50] == 1);
	assert(((double*)coding_tmps)[9+51] == 0.5);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == 0.5);
	assert(((double*)coding_tmps)[9+54] == 1);
	assert(((double*)coding_tmps)[9+55] == 0.5);
	assert(((double*)coding_tmps)[9+56] == 0.5);
	assert(((double*)coding_tmps)[9+57] == 1);
	assert(((double*)coding_tmps)[9+58] == 1);
	assert(((double*)coding_tmps)[9+59] == 0.5);
	assert(((double*)coding_tmps)[9+60] == 1);
	assert(((double*)coding_tmps)[9+61] == 0.5);
	assert(((double*)coding_tmps)[9+62] == 1);
	assert(((double*)coding_tmps)[9+63] == 0.5);
	assert(((double*)coding_tmps)[9+64] == 1);
	assert(((double*)coding_tmps)[9+65] == 0.5);
	assert(((double*)coding_tmps)[9+66] == 1);
	assert(((double*)coding_tmps)[9+67] == 0.5);
       	assert(((double*)coding_tmps)[9+68] == 1);
	assert(((double*)coding_tmps)[9+69] == 0.5);
	assert(((double*)coding_tmps)[9+70] == 1);
	assert(((double*)coding_tmps)[9+71] == 0.5);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With GlobalOrder nonlinearity, NoSign polarity split and SumAbs reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = GLOBAL_ORDER;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NO_SIGN;
        enum reduce_type          reduce_type = SUM_ABS;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 0.5 + 0.5 + 0.5);
	assert(o_coeffs[1] == 1 + 1);
	assert(o_coeffs[2] == 1 + 1);
	assert(o_coeffs[3] == 0.5 + 1 + 1);
	assert(o_coeffs[4] == 0.5 + 1 + 1);
	assert(o_coeffs[5] == 1 + 1);
	assert(o_coeffs[6] == 0.5 + 1);
	assert(o_coeffs[7] == 1 + 1);
	assert(o_coeffs[8] == 1 + 1);
	assert(o_coeffs[9] == 1);
	assert(o_coeffs[10] == 1 + 1);
	assert(o_coeffs[11] == 1 + 1);
	assert(o_coeffs[12] == 1 + 1);
	assert(o_coeffs[13] == 0.5 + 1 + 1);
	assert(o_coeffs[14] == 1 + 1);
	assert(o_coeffs[15] == 0.5);
	assert(o_coeffs[16] == 1 + 1);
	assert(o_coeffs[17] == 1 + 1);
	assert(o_coeffs[18] == 0.5 + 0.5);
	assert(o_coeffs[19] == 0.5 + 0.5 + 0.5 + 0.5);
	assert(o_coeffs[20] == 0.5 + 0.5);
	assert(o_coeffs[21] == 0.5 + 0.5 + 0.5 + 0.5);
	assert(o_coeffs[22] == 0.5 + 0.5);
	assert(o_coeffs[23] == 0.5 + 0.5 + 0.5 + 0.5);
	assert(o_coeffs[24] == 0.5 + 0.5);
	assert(o_coeffs[25] == 0.5);
	assert(o_coeffs[26] == 1 + 1);
	assert(o_coeffs[27] == 0.5 + 0.5);
	assert(o_coeffs[28] == 0.5 + 0.5);
	assert(o_coeffs[29] == 0.5);
	assert(o_coeffs[30] == 0.5 + 1);
	assert(o_coeffs[31] == 1 + 1);
	assert(o_coeffs[32] == 0.5 + 1);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == 0.5);
	assert(((double*)coding_tmps)[9+2] == 1);
	assert(((double*)coding_tmps)[9+3] == 0.5);
	assert(((double*)coding_tmps)[9+4] == 0.5);
	assert(((double*)coding_tmps)[9+5] == 1);
	assert(((double*)coding_tmps)[9+6] == 1);
	assert(((double*)coding_tmps)[9+7] == 0.5);
	assert(((double*)coding_tmps)[9+8] == 1);
	assert(((double*)coding_tmps)[9+9] == 0.5);
	assert(((double*)coding_tmps)[9+10] == 1);
	assert(((double*)coding_tmps)[9+11] == 0.5);
	assert(((double*)coding_tmps)[9+12] == 1);
	assert(((double*)coding_tmps)[9+13] == 0.5);
	assert(((double*)coding_tmps)[9+14] == 1);
	assert(((double*)coding_tmps)[9+15] == 0.5);
	assert(((double*)coding_tmps)[9+16] == 1);
	assert(((double*)coding_tmps)[9+17] == 0.5);
	assert(((double*)coding_tmps)[9+18] == 1);
	assert(((double*)coding_tmps)[9+19] == 0.5);
	assert(((double*)coding_tmps)[9+20] == 1);
	assert(((double*)coding_tmps)[9+21] == 0.5);
	assert(((double*)coding_tmps)[9+22] == 1);
	assert(((double*)coding_tmps)[9+23] == 0.5);
	assert(((double*)coding_tmps)[9+24] == 0.5);
	assert(((double*)coding_tmps)[9+25] == 1);
	assert(((double*)coding_tmps)[9+26] == 0.5);
	assert(((double*)coding_tmps)[9+27] == 1);
	assert(((double*)coding_tmps)[9+28] == 0.5);
	assert(((double*)coding_tmps)[9+29] == 1);
	assert(((double*)coding_tmps)[9+30] == 0.5);
	assert(((double*)coding_tmps)[9+31] == 1);
	assert(((double*)coding_tmps)[9+32] == 1);
	assert(((double*)coding_tmps)[9+33] == 0.5);
	assert(((double*)coding_tmps)[9+34] == 1);
	assert(((double*)coding_tmps)[9+35] == 0.5);
	assert(((double*)coding_tmps)[9+36] == 1);
	assert(((double*)coding_tmps)[9+37] == 0.5);
	assert(((double*)coding_tmps)[9+38] == 1);
	assert(((double*)coding_tmps)[9+39] == 0.5);
	assert(((double*)coding_tmps)[9+40] == 1);
	assert(((double*)coding_tmps)[9+41] == 0.5);
	assert(((double*)coding_tmps)[9+42] == 1);
	assert(((double*)coding_tmps)[9+43] == 0.5);
	assert(((double*)coding_tmps)[9+44] == 1);
	assert(((double*)coding_tmps)[9+45] == 0.5);
	assert(((double*)coding_tmps)[9+46] == 1);
	assert(((double*)coding_tmps)[9+47] == 0.5);
	assert(((double*)coding_tmps)[9+48] == 0.5);
	assert(((double*)coding_tmps)[9+49] == 1);
	assert(((double*)coding_tmps)[9+50] == 1);
	assert(((double*)coding_tmps)[9+51] == 0.5);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == 0.5);
	assert(((double*)coding_tmps)[9+54] == 1);
	assert(((double*)coding_tmps)[9+55] == 0.5);
	assert(((double*)coding_tmps)[9+56] == 0.5);
	assert(((double*)coding_tmps)[9+57] == 1);
	assert(((double*)coding_tmps)[9+58] == 1);
	assert(((double*)coding_tmps)[9+59] == 0.5);
	assert(((double*)coding_tmps)[9+60] == 1);
	assert(((double*)coding_tmps)[9+61] == 0.5);
	assert(((double*)coding_tmps)[9+62] == 1);
	assert(((double*)coding_tmps)[9+63] == 0.5);
	assert(((double*)coding_tmps)[9+64] == 1);
	assert(((double*)coding_tmps)[9+65] == 0.5);
	assert(((double*)coding_tmps)[9+66] == 1);
	assert(((double*)coding_tmps)[9+67] == 0.5);
       	assert(((double*)coding_tmps)[9+68] == 1);
	assert(((double*)coding_tmps)[9+69] == 0.5);
	assert(((double*)coding_tmps)[9+70] == 1);
	assert(((double*)coding_tmps)[9+71] == 0.5);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With GlobalOrder nonlinearity, NoSign polarity split and SumSqr reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = GLOBAL_ORDER;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = NO_SIGN;
        enum reduce_type          reduce_type = SUM_SQR;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 0.5*0.5 + 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[1] == 1*1 + 1*1);
	assert(o_coeffs[2] == 1*1 + 1*1);
	assert(o_coeffs[3] == 0.5*0.5 + 1*1 + 1*1);
	assert(o_coeffs[4] == 0.5*0.5 + 1*1 + 1*1);
	assert(o_coeffs[5] == 1*1 + 1*1);
	assert(o_coeffs[6] == 0.5*0.5 + 1*1);
	assert(o_coeffs[7] == 1*1 + 1*1);
	assert(o_coeffs[8] == 1*1 + 1*1);
	assert(o_coeffs[9] == 1*1);
	assert(o_coeffs[10] == 1*1 + 1*1);
	assert(o_coeffs[11] == 1*1 + 1*1);
	assert(o_coeffs[12] == 1*1 + 1*1);
	assert(o_coeffs[13] == 0.5*0.5 + 1*1 + 1*1);
	assert(o_coeffs[14] == 1*1 + 1*1);
	assert(o_coeffs[15] == 0.5*0.5);
	assert(o_coeffs[16] == 1*1 + 1*1);
	assert(o_coeffs[17] == 1*1 + 1*1);
	assert(o_coeffs[18] == 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[19] == 0.5*0.5 + 0.5*0.5 + 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[20] == 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[21] == 0.5*0.5 + 0.5*0.5 + 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[22] == 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[23] == 0.5*0.5 + 0.5*0.5 + 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[24] == 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[25] == 0.5*0.5);
	assert(o_coeffs[26] == 1*1 + 1*1);
	assert(o_coeffs[27] == 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[28] == 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[29] == 0.5*0.5);
	assert(o_coeffs[30] == 0.5*0.5 + 1*1);
	assert(o_coeffs[31] == 1*1 + 1*1);
	assert(o_coeffs[32] == 0.5*0.5 + 1*1);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == 0.5);
	assert(((double*)coding_tmps)[9+2] == 1);
	assert(((double*)coding_tmps)[9+3] == 0.5);
	assert(((double*)coding_tmps)[9+4] == 0.5);
	assert(((double*)coding_tmps)[9+5] == 1);
	assert(((double*)coding_tmps)[9+6] == 1);
	assert(((double*)coding_tmps)[9+7] == 0.5);
	assert(((double*)coding_tmps)[9+8] == 1);
	assert(((double*)coding_tmps)[9+9] == 0.5);
	assert(((double*)coding_tmps)[9+10] == 1);
	assert(((double*)coding_tmps)[9+11] == 0.5);
	assert(((double*)coding_tmps)[9+12] == 1);
	assert(((double*)coding_tmps)[9+13] == 0.5);
	assert(((double*)coding_tmps)[9+14] == 1);
	assert(((double*)coding_tmps)[9+15] == 0.5);
	assert(((double*)coding_tmps)[9+16] == 1);
	assert(((double*)coding_tmps)[9+17] == 0.5);
	assert(((double*)coding_tmps)[9+18] == 1);
	assert(((double*)coding_tmps)[9+19] == 0.5);
	assert(((double*)coding_tmps)[9+20] == 1);
	assert(((double*)coding_tmps)[9+21] == 0.5);
	assert(((double*)coding_tmps)[9+22] == 1);
	assert(((double*)coding_tmps)[9+23] == 0.5);
	assert(((double*)coding_tmps)[9+24] == 0.5);
	assert(((double*)coding_tmps)[9+25] == 1);
	assert(((double*)coding_tmps)[9+26] == 0.5);
	assert(((double*)coding_tmps)[9+27] == 1);
	assert(((double*)coding_tmps)[9+28] == 0.5);
	assert(((double*)coding_tmps)[9+29] == 1);
	assert(((double*)coding_tmps)[9+30] == 0.5);
	assert(((double*)coding_tmps)[9+31] == 1);
	assert(((double*)coding_tmps)[9+32] == 1);
	assert(((double*)coding_tmps)[9+33] == 0.5);
	assert(((double*)coding_tmps)[9+34] == 1);
	assert(((double*)coding_tmps)[9+35] == 0.5);
	assert(((double*)coding_tmps)[9+36] == 1);
	assert(((double*)coding_tmps)[9+37] == 0.5);
	assert(((double*)coding_tmps)[9+38] == 1);
	assert(((double*)coding_tmps)[9+39] == 0.5);
	assert(((double*)coding_tmps)[9+40] == 1);
	assert(((double*)coding_tmps)[9+41] == 0.5);
	assert(((double*)coding_tmps)[9+42] == 1);
	assert(((double*)coding_tmps)[9+43] == 0.5);
	assert(((double*)coding_tmps)[9+44] == 1);
	assert(((double*)coding_tmps)[9+45] == 0.5);
	assert(((double*)coding_tmps)[9+46] == 1);
	assert(((double*)coding_tmps)[9+47] == 0.5);
	assert(((double*)coding_tmps)[9+48] == 0.5);
	assert(((double*)coding_tmps)[9+49] == 1);
	assert(((double*)coding_tmps)[9+50] == 1);
	assert(((double*)coding_tmps)[9+51] == 0.5);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == 0.5);
	assert(((double*)coding_tmps)[9+54] == 1);
	assert(((double*)coding_tmps)[9+55] == 0.5);
	assert(((double*)coding_tmps)[9+56] == 0.5);
	assert(((double*)coding_tmps)[9+57] == 1);
	assert(((double*)coding_tmps)[9+58] == 1);
	assert(((double*)coding_tmps)[9+59] == 0.5);
	assert(((double*)coding_tmps)[9+60] == 1);
	assert(((double*)coding_tmps)[9+61] == 0.5);
	assert(((double*)coding_tmps)[9+62] == 1);
	assert(((double*)coding_tmps)[9+63] == 0.5);
	assert(((double*)coding_tmps)[9+64] == 1);
	assert(((double*)coding_tmps)[9+65] == 0.5);
	assert(((double*)coding_tmps)[9+66] == 1);
	assert(((double*)coding_tmps)[9+67] == 0.5);
       	assert(((double*)coding_tmps)[9+68] == 1);
	assert(((double*)coding_tmps)[9+69] == 0.5);
	assert(((double*)coding_tmps)[9+70] == 1);
	assert(((double*)coding_tmps)[9+71] == 0.5);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With GlobalOrder nonlinearity, KeepSign polarity split and Subsample reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = GLOBAL_ORDER;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = KEEP_SIGN;
        enum reduce_type          reduce_type = SUBSAMPLE;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 18);
	assert(o_coeffs[0] == 0.5);
	assert(o_coeffs[1] == 1);
	assert(o_coeffs[2] == 1);
	assert(o_coeffs[3] == 0.5);
	assert(o_coeffs[4] == 0.5);
	assert(o_coeffs[5] == 1);
	assert(o_coeffs[6] == 1);
	assert(o_coeffs[7] == 1);
	assert(o_coeffs[8] == 1);
	assert(o_coeffs[9] == 1);
	assert(o_coeffs[10] == -0.5);
	assert(o_coeffs[11] == -0.5);
	assert(o_coeffs[12] == -0.5);
	assert(o_coeffs[13] == -0.5);
	assert(o_coeffs[14] == -0.5);
	assert(o_coeffs[15] == -0.5);
	assert(o_coeffs[16] == -1);
	assert(o_coeffs[17] == -1);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 16);
	assert(o_coeffs_idx[6] == 17);
	assert(o_coeffs_idx[7] == 18);
	assert(o_coeffs_idx[8] == 19);
	assert(o_coeffs_idx[9] == 20);
	assert(o_coeffs_idx[10] == 54+28);
	assert(o_coeffs_idx[11] == 54+29);
	assert(o_coeffs_idx[12] == 54+31);
	assert(o_coeffs_idx[13] == 54+32);
	assert(o_coeffs_idx[14] == 54+35);
	assert(o_coeffs_idx[15] == 54+45);
	assert(o_coeffs_idx[16] == 54+48);
	assert(o_coeffs_idx[17] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -0.5);
	assert(((double*)coding_tmps)[9+2] == 1);
	assert(((double*)coding_tmps)[9+3] == -0.5);
	assert(((double*)coding_tmps)[9+4] == 0.5);
	assert(((double*)coding_tmps)[9+5] == -1);
	assert(((double*)coding_tmps)[9+6] == 1);
	assert(((double*)coding_tmps)[9+7] == -0.5);
	assert(((double*)coding_tmps)[9+8] == 1);
	assert(((double*)coding_tmps)[9+9] == -0.5);
	assert(((double*)coding_tmps)[9+10] == 1);
	assert(((double*)coding_tmps)[9+11] == -0.5);
	assert(((double*)coding_tmps)[9+12] == 1);
	assert(((double*)coding_tmps)[9+13] == 0.5);
	assert(((double*)coding_tmps)[9+14] == 1);
	assert(((double*)coding_tmps)[9+15] == -0.5);
	assert(((double*)coding_tmps)[9+16] == 1);
	assert(((double*)coding_tmps)[9+17] == -0.5);
	assert(((double*)coding_tmps)[9+18] == 1);
	assert(((double*)coding_tmps)[9+19] == -0.5);
	assert(((double*)coding_tmps)[9+20] == 1);
	assert(((double*)coding_tmps)[9+21] == -0.5);
	assert(((double*)coding_tmps)[9+22] == 1);
	assert(((double*)coding_tmps)[9+23] == -0.5);
	assert(((double*)coding_tmps)[9+24] == 0.5);
	assert(((double*)coding_tmps)[9+25] == -1);
	assert(((double*)coding_tmps)[9+26] == 0.5);
	assert(((double*)coding_tmps)[9+27] == -1);
	assert(((double*)coding_tmps)[9+28] == 0.5);
	assert(((double*)coding_tmps)[9+29] == -1);
	assert(((double*)coding_tmps)[9+30] == 0.5);
	assert(((double*)coding_tmps)[9+31] == -1);
	assert(((double*)coding_tmps)[9+32] == 1);
	assert(((double*)coding_tmps)[9+33] == -0.5);
	assert(((double*)coding_tmps)[9+34] == 1);
	assert(((double*)coding_tmps)[9+35] == -0.5);
	assert(((double*)coding_tmps)[9+36] == 1);
	assert(((double*)coding_tmps)[9+37] == -0.5);
	assert(((double*)coding_tmps)[9+38] == 1);
	assert(((double*)coding_tmps)[9+39] == -0.5);
	assert(((double*)coding_tmps)[9+40] == 1);
	assert(((double*)coding_tmps)[9+41] == -0.5);
	assert(((double*)coding_tmps)[9+42] == 1);
	assert(((double*)coding_tmps)[9+43] == -0.5);
	assert(((double*)coding_tmps)[9+44] == 1);
	assert(((double*)coding_tmps)[9+45] == -0.5);
	assert(((double*)coding_tmps)[9+46] == 1);
	assert(((double*)coding_tmps)[9+47] == -0.5);
	assert(((double*)coding_tmps)[9+48] == 0.5);
	assert(((double*)coding_tmps)[9+49] == -1);
	assert(((double*)coding_tmps)[9+50] == 1);
	assert(((double*)coding_tmps)[9+51] == -0.5);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -0.5);
	assert(((double*)coding_tmps)[9+54] == 1);
	assert(((double*)coding_tmps)[9+55] == -0.5);
	assert(((double*)coding_tmps)[9+56] == 0.5);
	assert(((double*)coding_tmps)[9+57] == 1);
	assert(((double*)coding_tmps)[9+58] == 1);
	assert(((double*)coding_tmps)[9+59] == -0.5);
	assert(((double*)coding_tmps)[9+60] == 1);
	assert(((double*)coding_tmps)[9+61] == -0.5);
	assert(((double*)coding_tmps)[9+62] == 1);
	assert(((double*)coding_tmps)[9+63] == -0.5);
	assert(((double*)coding_tmps)[9+64] == 1);
	assert(((double*)coding_tmps)[9+65] == -0.5);
	assert(((double*)coding_tmps)[9+66] == 1);
	assert(((double*)coding_tmps)[9+67] == -0.5);
       	assert(((double*)coding_tmps)[9+68] == 1);
	assert(((double*)coding_tmps)[9+69] == -0.5);
	assert(((double*)coding_tmps)[9+70] == 1);
	assert(((double*)coding_tmps)[9+71] == -0.5);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 0);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 0);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With GlobalOrder nonlinearity, KeepSign polarity split and MaxNoSign reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = GLOBAL_ORDER;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = KEEP_SIGN;
        enum reduce_type          reduce_type = MAX_NO_SIGN;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 0.5);
	assert(o_coeffs[1] == 1);
	assert(o_coeffs[2] == 1);
	assert(o_coeffs[3] == 1);
	assert(o_coeffs[4] == 1);
	assert(o_coeffs[5] == 1);
	assert(o_coeffs[6] == 1);
	assert(o_coeffs[7] == 1);
	assert(o_coeffs[8] == 1);
	assert(o_coeffs[9] == 1);
	assert(o_coeffs[10] == 1);
	assert(o_coeffs[11] == 1);
	assert(o_coeffs[12] == 1);
	assert(o_coeffs[13] == 1);
	assert(o_coeffs[14] == 1);
	assert(o_coeffs[15] == 0.5);
	assert(o_coeffs[16] == 1);
	assert(o_coeffs[17] == 1);
	assert(o_coeffs[18] == 0.5);
	assert(o_coeffs[19] == 0.5);
	assert(o_coeffs[20] == 0.5);
	assert(o_coeffs[21] == 0.5);
	assert(o_coeffs[22] == 0.5);
	assert(o_coeffs[23] == 0.5);
	assert(o_coeffs[24] == 0.5);
	assert(o_coeffs[25] == 0.5);
	assert(o_coeffs[26] == 1);
	assert(o_coeffs[27] == 0.5);
	assert(o_coeffs[28] == 0.5);
	assert(o_coeffs[29] == 0.5);
	assert(o_coeffs[30] == 1);
	assert(o_coeffs[31] == 1);
	assert(o_coeffs[32] == 1);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -0.5);
	assert(((double*)coding_tmps)[9+2] == 1);
	assert(((double*)coding_tmps)[9+3] == -0.5);
	assert(((double*)coding_tmps)[9+4] == 0.5);
	assert(((double*)coding_tmps)[9+5] == -1);
	assert(((double*)coding_tmps)[9+6] == 1);
	assert(((double*)coding_tmps)[9+7] == -0.5);
	assert(((double*)coding_tmps)[9+8] == 1);
	assert(((double*)coding_tmps)[9+9] == -0.5);
	assert(((double*)coding_tmps)[9+10] == 1);
	assert(((double*)coding_tmps)[9+11] == -0.5);
	assert(((double*)coding_tmps)[9+12] == 1);
	assert(((double*)coding_tmps)[9+13] == 0.5);
	assert(((double*)coding_tmps)[9+14] == 1);
	assert(((double*)coding_tmps)[9+15] == -0.5);
	assert(((double*)coding_tmps)[9+16] == 1);
	assert(((double*)coding_tmps)[9+17] == -0.5);
	assert(((double*)coding_tmps)[9+18] == 1);
	assert(((double*)coding_tmps)[9+19] == -0.5);
	assert(((double*)coding_tmps)[9+20] == 1);
	assert(((double*)coding_tmps)[9+21] == -0.5);
	assert(((double*)coding_tmps)[9+22] == 1);
	assert(((double*)coding_tmps)[9+23] == -0.5);
	assert(((double*)coding_tmps)[9+24] == 0.5);
	assert(((double*)coding_tmps)[9+25] == -1);
	assert(((double*)coding_tmps)[9+26] == 0.5);
	assert(((double*)coding_tmps)[9+27] == -1);
	assert(((double*)coding_tmps)[9+28] == 0.5);
	assert(((double*)coding_tmps)[9+29] == -1);
	assert(((double*)coding_tmps)[9+30] == 0.5);
	assert(((double*)coding_tmps)[9+31] == -1);
	assert(((double*)coding_tmps)[9+32] == 1);
	assert(((double*)coding_tmps)[9+33] == -0.5);
	assert(((double*)coding_tmps)[9+34] == 1);
	assert(((double*)coding_tmps)[9+35] == -0.5);
	assert(((double*)coding_tmps)[9+36] == 1);
	assert(((double*)coding_tmps)[9+37] == -0.5);
	assert(((double*)coding_tmps)[9+38] == 1);
	assert(((double*)coding_tmps)[9+39] == -0.5);
	assert(((double*)coding_tmps)[9+40] == 1);
	assert(((double*)coding_tmps)[9+41] == -0.5);
	assert(((double*)coding_tmps)[9+42] == 1);
	assert(((double*)coding_tmps)[9+43] == -0.5);
	assert(((double*)coding_tmps)[9+44] == 1);
	assert(((double*)coding_tmps)[9+45] == -0.5);
	assert(((double*)coding_tmps)[9+46] == 1);
	assert(((double*)coding_tmps)[9+47] == -0.5);
	assert(((double*)coding_tmps)[9+48] == 0.5);
	assert(((double*)coding_tmps)[9+49] == -1);
	assert(((double*)coding_tmps)[9+50] == 1);
	assert(((double*)coding_tmps)[9+51] == -0.5);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -0.5);
	assert(((double*)coding_tmps)[9+54] == 1);
	assert(((double*)coding_tmps)[9+55] == -0.5);
	assert(((double*)coding_tmps)[9+56] == 0.5);
	assert(((double*)coding_tmps)[9+57] == 1);
	assert(((double*)coding_tmps)[9+58] == 1);
	assert(((double*)coding_tmps)[9+59] == -0.5);
	assert(((double*)coding_tmps)[9+60] == 1);
	assert(((double*)coding_tmps)[9+61] == -0.5);
	assert(((double*)coding_tmps)[9+62] == 1);
	assert(((double*)coding_tmps)[9+63] == -0.5);
	assert(((double*)coding_tmps)[9+64] == 1);
	assert(((double*)coding_tmps)[9+65] == -0.5);
	assert(((double*)coding_tmps)[9+66] == 1);
	assert(((double*)coding_tmps)[9+67] == -0.5);
       	assert(((double*)coding_tmps)[9+68] == 1);
	assert(((double*)coding_tmps)[9+69] == -0.5);
	assert(((double*)coding_tmps)[9+70] == 1);
	assert(((double*)coding_tmps)[9+71] == -0.5);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With GlobalOrder nonlinearity, KeepSign polarity split and MaxKeepSign reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = GLOBAL_ORDER;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = KEEP_SIGN;
        enum reduce_type          reduce_type = MAX_KEEP_SIGN;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 0.5);
	assert(o_coeffs[1] == 1);
	assert(o_coeffs[2] == 1);
	assert(o_coeffs[3] == 1);
	assert(o_coeffs[4] == 1);
	assert(o_coeffs[5] == 1);
	assert(o_coeffs[6] == 1);
	assert(o_coeffs[7] == 1);
	assert(o_coeffs[8] == 1);
	assert(o_coeffs[9] == 1);
	assert(o_coeffs[10] == 1);
	assert(o_coeffs[11] == 1);
	assert(o_coeffs[12] == 1);
	assert(o_coeffs[13] == 1);
	assert(o_coeffs[14] == 1);
	assert(o_coeffs[15] == 0.5);
	assert(o_coeffs[16] == 1);
	assert(o_coeffs[17] == 1);
	assert(o_coeffs[18] == -0.5);
	assert(o_coeffs[19] == -0.5);
	assert(o_coeffs[20] == -0.5);
	assert(o_coeffs[21] == -0.5);
	assert(o_coeffs[22] == -0.5);
	assert(o_coeffs[23] == -0.5);
	assert(o_coeffs[24] == -0.5);
	assert(o_coeffs[25] == -0.5);
	assert(o_coeffs[26] == -1);
	assert(o_coeffs[27] == -0.5);
	assert(o_coeffs[28] == -0.5);
	assert(o_coeffs[29] == -0.5);
	assert(o_coeffs[30] == -1);
	assert(o_coeffs[31] == -1);
	assert(o_coeffs[32] == -1);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -0.5);
	assert(((double*)coding_tmps)[9+2] == 1);
	assert(((double*)coding_tmps)[9+3] == -0.5);
	assert(((double*)coding_tmps)[9+4] == 0.5);
	assert(((double*)coding_tmps)[9+5] == -1);
	assert(((double*)coding_tmps)[9+6] == 1);
	assert(((double*)coding_tmps)[9+7] == -0.5);
	assert(((double*)coding_tmps)[9+8] == 1);
	assert(((double*)coding_tmps)[9+9] == -0.5);
	assert(((double*)coding_tmps)[9+10] == 1);
	assert(((double*)coding_tmps)[9+11] == -0.5);
	assert(((double*)coding_tmps)[9+12] == 1);
	assert(((double*)coding_tmps)[9+13] == 0.5);
	assert(((double*)coding_tmps)[9+14] == 1);
	assert(((double*)coding_tmps)[9+15] == -0.5);
	assert(((double*)coding_tmps)[9+16] == 1);
	assert(((double*)coding_tmps)[9+17] == -0.5);
	assert(((double*)coding_tmps)[9+18] == 1);
	assert(((double*)coding_tmps)[9+19] == -0.5);
	assert(((double*)coding_tmps)[9+20] == 1);
	assert(((double*)coding_tmps)[9+21] == -0.5);
	assert(((double*)coding_tmps)[9+22] == 1);
	assert(((double*)coding_tmps)[9+23] == -0.5);
	assert(((double*)coding_tmps)[9+24] == 0.5);
	assert(((double*)coding_tmps)[9+25] == -1);
	assert(((double*)coding_tmps)[9+26] == 0.5);
	assert(((double*)coding_tmps)[9+27] == -1);
	assert(((double*)coding_tmps)[9+28] == 0.5);
	assert(((double*)coding_tmps)[9+29] == -1);
	assert(((double*)coding_tmps)[9+30] == 0.5);
	assert(((double*)coding_tmps)[9+31] == -1);
	assert(((double*)coding_tmps)[9+32] == 1);
	assert(((double*)coding_tmps)[9+33] == -0.5);
	assert(((double*)coding_tmps)[9+34] == 1);
	assert(((double*)coding_tmps)[9+35] == -0.5);
	assert(((double*)coding_tmps)[9+36] == 1);
	assert(((double*)coding_tmps)[9+37] == -0.5);
	assert(((double*)coding_tmps)[9+38] == 1);
	assert(((double*)coding_tmps)[9+39] == -0.5);
	assert(((double*)coding_tmps)[9+40] == 1);
	assert(((double*)coding_tmps)[9+41] == -0.5);
	assert(((double*)coding_tmps)[9+42] == 1);
	assert(((double*)coding_tmps)[9+43] == -0.5);
	assert(((double*)coding_tmps)[9+44] == 1);
	assert(((double*)coding_tmps)[9+45] == -0.5);
	assert(((double*)coding_tmps)[9+46] == 1);
	assert(((double*)coding_tmps)[9+47] == -0.5);
	assert(((double*)coding_tmps)[9+48] == 0.5);
	assert(((double*)coding_tmps)[9+49] == -1);
	assert(((double*)coding_tmps)[9+50] == 1);
	assert(((double*)coding_tmps)[9+51] == -0.5);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -0.5);
	assert(((double*)coding_tmps)[9+54] == 1);
	assert(((double*)coding_tmps)[9+55] == -0.5);
	assert(((double*)coding_tmps)[9+56] == 0.5);
	assert(((double*)coding_tmps)[9+57] == 1);
	assert(((double*)coding_tmps)[9+58] == 1);
	assert(((double*)coding_tmps)[9+59] == -0.5);
	assert(((double*)coding_tmps)[9+60] == 1);
	assert(((double*)coding_tmps)[9+61] == -0.5);
	assert(((double*)coding_tmps)[9+62] == 1);
	assert(((double*)coding_tmps)[9+63] == -0.5);
	assert(((double*)coding_tmps)[9+64] == 1);
	assert(((double*)coding_tmps)[9+65] == -0.5);
	assert(((double*)coding_tmps)[9+66] == 1);
	assert(((double*)coding_tmps)[9+67] == -0.5);
       	assert(((double*)coding_tmps)[9+68] == 1);
	assert(((double*)coding_tmps)[9+69] == -0.5);
	assert(((double*)coding_tmps)[9+70] == 1);
	assert(((double*)coding_tmps)[9+71] == -0.5);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With GlobalOrder nonlinearity, KeepSign polarity split and SumAbs reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = GLOBAL_ORDER;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = KEEP_SIGN;
        enum reduce_type          reduce_type = SUM_ABS;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 0.5 + 0.5 + 0.5);
	assert(o_coeffs[1] == 1 + 1);
	assert(o_coeffs[2] == 1 + 1);
	assert(o_coeffs[3] == 0.5 + 1 + 1);
	assert(o_coeffs[4] == 0.5 + 1 + 1);
	assert(o_coeffs[5] == 1 + 1);
	assert(o_coeffs[6] == 0.5 + 1);
	assert(o_coeffs[7] == 1 + 1);
	assert(o_coeffs[8] == 1 + 1);
	assert(o_coeffs[9] == 1);
	assert(o_coeffs[10] == 1 + 1);
	assert(o_coeffs[11] == 1 + 1);
	assert(o_coeffs[12] == 1 + 1);
	assert(o_coeffs[13] == 0.5 + 1 + 1);
	assert(o_coeffs[14] == 1 + 1);
	assert(o_coeffs[15] == 0.5);
	assert(o_coeffs[16] == 1 + 1);
	assert(o_coeffs[17] == 1 + 1);
	assert(o_coeffs[18] == 0.5 + 0.5);
	assert(o_coeffs[19] == 0.5 + 0.5 + 0.5 + 0.5);
	assert(o_coeffs[20] == 0.5 + 0.5);
	assert(o_coeffs[21] == 0.5 + 0.5 + 0.5 + 0.5);
	assert(o_coeffs[22] == 0.5 + 0.5);
	assert(o_coeffs[23] == 0.5 + 0.5 + 0.5 + 0.5);
	assert(o_coeffs[24] == 0.5 + 0.5);
	assert(o_coeffs[25] == 0.5);
	assert(o_coeffs[26] == 1 + 1);
	assert(o_coeffs[27] == 0.5 + 0.5);
	assert(o_coeffs[28] == 0.5 + 0.5);
	assert(o_coeffs[29] == 0.5);
	assert(o_coeffs[30] == 0.5 + 1);
	assert(o_coeffs[31] == 1 + 1);
	assert(o_coeffs[32] == 0.5 + 1);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -0.5);
	assert(((double*)coding_tmps)[9+2] == 1);
	assert(((double*)coding_tmps)[9+3] == -0.5);
	assert(((double*)coding_tmps)[9+4] == 0.5);
	assert(((double*)coding_tmps)[9+5] == -1);
	assert(((double*)coding_tmps)[9+6] == 1);
	assert(((double*)coding_tmps)[9+7] == -0.5);
	assert(((double*)coding_tmps)[9+8] == 1);
	assert(((double*)coding_tmps)[9+9] == -0.5);
	assert(((double*)coding_tmps)[9+10] == 1);
	assert(((double*)coding_tmps)[9+11] == -0.5);
	assert(((double*)coding_tmps)[9+12] == 1);
	assert(((double*)coding_tmps)[9+13] == 0.5);
	assert(((double*)coding_tmps)[9+14] == 1);
	assert(((double*)coding_tmps)[9+15] == -0.5);
	assert(((double*)coding_tmps)[9+16] == 1);
	assert(((double*)coding_tmps)[9+17] == -0.5);
	assert(((double*)coding_tmps)[9+18] == 1);
	assert(((double*)coding_tmps)[9+19] == -0.5);
	assert(((double*)coding_tmps)[9+20] == 1);
	assert(((double*)coding_tmps)[9+21] == -0.5);
	assert(((double*)coding_tmps)[9+22] == 1);
	assert(((double*)coding_tmps)[9+23] == -0.5);
	assert(((double*)coding_tmps)[9+24] == 0.5);
	assert(((double*)coding_tmps)[9+25] == -1);
	assert(((double*)coding_tmps)[9+26] == 0.5);
	assert(((double*)coding_tmps)[9+27] == -1);
	assert(((double*)coding_tmps)[9+28] == 0.5);
	assert(((double*)coding_tmps)[9+29] == -1);
	assert(((double*)coding_tmps)[9+30] == 0.5);
	assert(((double*)coding_tmps)[9+31] == -1);
	assert(((double*)coding_tmps)[9+32] == 1);
	assert(((double*)coding_tmps)[9+33] == -0.5);
	assert(((double*)coding_tmps)[9+34] == 1);
	assert(((double*)coding_tmps)[9+35] == -0.5);
	assert(((double*)coding_tmps)[9+36] == 1);
	assert(((double*)coding_tmps)[9+37] == -0.5);
	assert(((double*)coding_tmps)[9+38] == 1);
	assert(((double*)coding_tmps)[9+39] == -0.5);
	assert(((double*)coding_tmps)[9+40] == 1);
	assert(((double*)coding_tmps)[9+41] == -0.5);
	assert(((double*)coding_tmps)[9+42] == 1);
	assert(((double*)coding_tmps)[9+43] == -0.5);
	assert(((double*)coding_tmps)[9+44] == 1);
	assert(((double*)coding_tmps)[9+45] == -0.5);
	assert(((double*)coding_tmps)[9+46] == 1);
	assert(((double*)coding_tmps)[9+47] == -0.5);
	assert(((double*)coding_tmps)[9+48] == 0.5);
	assert(((double*)coding_tmps)[9+49] == -1);
	assert(((double*)coding_tmps)[9+50] == 1);
	assert(((double*)coding_tmps)[9+51] == -0.5);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -0.5);
	assert(((double*)coding_tmps)[9+54] == 1);
	assert(((double*)coding_tmps)[9+55] == -0.5);
	assert(((double*)coding_tmps)[9+56] == 0.5);
	assert(((double*)coding_tmps)[9+57] == 1);
	assert(((double*)coding_tmps)[9+58] == 1);
	assert(((double*)coding_tmps)[9+59] == -0.5);
	assert(((double*)coding_tmps)[9+60] == 1);
	assert(((double*)coding_tmps)[9+61] == -0.5);
	assert(((double*)coding_tmps)[9+62] == 1);
	assert(((double*)coding_tmps)[9+63] == -0.5);
	assert(((double*)coding_tmps)[9+64] == 1);
	assert(((double*)coding_tmps)[9+65] == -0.5);
	assert(((double*)coding_tmps)[9+66] == 1);
	assert(((double*)coding_tmps)[9+67] == -0.5);
       	assert(((double*)coding_tmps)[9+68] == 1);
	assert(((double*)coding_tmps)[9+69] == -0.5);
	assert(((double*)coding_tmps)[9+70] == 1);
	assert(((double*)coding_tmps)[9+71] == -0.5);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

    printf("    With GlobalOrder nonlinearity, KeepSign polarity split and SumSqr reduce.\n");

    {
	size_t                    o_coeff_count;
        double*                   o_coeffs;
        size_t*                   o_coeffs_idx;
        size_t                    geometry = 42;
        size_t                    row_count = 7;
        size_t                    col_count = 6;
        size_t                    patch_row_count = 3;
        size_t                    patch_col_count = 3;
        enum coding_type          coding_type = CORRELATION;
        size_t                    word_count = 6;
        double                    dict[] = { 1, 0, 0,-1, 0, 0,
					     1, 0, 0, 0,-1, 0,
					     1, 0, 0, 0, 0,-1,
					     0, 1, 0,-1, 0, 0,
					     0, 1, 0, 0,-1, 0,
					     0, 1, 0, 0, 0,-1,
					     0, 0, 1,-1, 0, 0,
					     0, 0, 1, 0,-1, 0,
					     0, 0, 1, 0, 0,-1};
        double                    dict_transp[] = { 1, 1, 1, 0, 0, 0, 0, 0, 0,
						    0, 0, 0, 1, 1, 1, 0, 0, 0,
						    0, 0, 0, 0, 0, 0, 1, 1, 1,
						   -1, 0, 0,-1, 0, 0,-1, 0, 0,
						    0,-1, 0, 0,-1, 0, 0,-1, 0,
						    0, 0,-1, 0, 0,-1, 0, 0,-1};
        double                    dict_x_dict_transp[] = {1,0,0,1,1,1,
							  0,1,0,1,1,1,
							  0,0,1,1,1,1,
							  1,1,1,1,0,0,
							  1,1,1,0,1,0,
							  1,1,1,0,0,1};
        size_t                    coeff_count = 2;
        enum nonlinear_type       nonlinear_type = GLOBAL_ORDER;
	double                    nonlinear_modulator[] = {1,0.5};
        enum polarity_split_type  polarity_split_type = KEEP_SIGN;
        enum reduce_type          reduce_type = SUM_SQR;
        size_t                    reduce_spread = 2;
        double                    observation[] = {0,0,0,0,0,0,0,
						   0,1,1,1,1,1,0,
						   0,1,0,1,0,0,0,
						   0,1,0,1,0,0,0,
						   0,1,1,1,1,1,0,
						   0,0,0,0,0,0,0};
        char*                     coding_tmps;
	size_t                    new_geometry;

	new_geometry = code_image_new_geometry(row_count,col_count,word_count,polarity_split_type,reduce_spread);

	o_coeffs = (double*)malloc(new_geometry * sizeof(double));
	o_coeffs_idx = (size_t*)malloc(new_geometry * sizeof(size_t));
        coding_tmps = (char*)malloc(code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));
	memset(coding_tmps,0,code_image_coding_tmps_length(row_count,col_count,patch_row_count,patch_col_count,coding_type,word_count,coeff_count,reduce_spread));

        code_image(&o_coeff_count,o_coeffs,o_coeffs_idx,geometry,row_count,col_count,patch_row_count,patch_col_count,
                   coding_type,word_count,dict,dict_transp,dict_x_dict_transp,coeff_count,NULL,
                   nonlinear_type,nonlinear_modulator,polarity_split_type,reduce_type,reduce_spread,observation,coding_tmps);

	assert(o_coeff_count == 33);
	assert(o_coeffs[0] == 0.5*0.5 + 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[1] == 1*1 + 1*1);
	assert(o_coeffs[2] == 1*1 + 1*1);
	assert(o_coeffs[3] == 0.5*0.5 + 1*1 + 1*1);
	assert(o_coeffs[4] == 0.5*0.5 + 1*1 + 1*1);
	assert(o_coeffs[5] == 1*1 + 1*1);
	assert(o_coeffs[6] == 0.5*0.5 + 1*1);
	assert(o_coeffs[7] == 1*1 + 1*1);
	assert(o_coeffs[8] == 1*1 + 1*1);
	assert(o_coeffs[9] == 1*1);
	assert(o_coeffs[10] == 1*1 + 1*1);
	assert(o_coeffs[11] == 1*1 + 1*1);
	assert(o_coeffs[12] == 1*1 + 1*1);
	assert(o_coeffs[13] == 0.5*0.5 + 1*1 + 1*1);
	assert(o_coeffs[14] == 1*1 + 1*1);
	assert(o_coeffs[15] == 0.5*0.5);
	assert(o_coeffs[16] == 1*1 + 1*1);
	assert(o_coeffs[17] == 1*1 + 1*1);
	assert(o_coeffs[18] == 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[19] == 0.5*0.5 + 0.5*0.5 + 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[20] == 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[21] == 0.5*0.5 + 0.5*0.5 + 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[22] == 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[23] == 0.5*0.5 + 0.5*0.5 + 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[24] == 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[25] == 0.5*0.5);
	assert(o_coeffs[26] == 1*1 + 1*1);
	assert(o_coeffs[27] == 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[28] == 0.5*0.5 + 0.5*0.5);
	assert(o_coeffs[29] == 0.5*0.5);
	assert(o_coeffs[30] == 0.5*0.5 + 1*1);
	assert(o_coeffs[31] == 1*1 + 1*1);
	assert(o_coeffs[32] == 0.5*0.5 + 1*1);
	assert(o_coeffs_idx[0] == 3);
	assert(o_coeffs_idx[1] == 4);
	assert(o_coeffs_idx[2] == 5);
	assert(o_coeffs_idx[3] == 6);
	assert(o_coeffs_idx[4] == 7);
	assert(o_coeffs_idx[5] == 8);
	assert(o_coeffs_idx[6] == 9);
	assert(o_coeffs_idx[7] == 10);
	assert(o_coeffs_idx[8] == 11);
	assert(o_coeffs_idx[9] == 15);
	assert(o_coeffs_idx[10] == 16);
	assert(o_coeffs_idx[11] == 17);
	assert(o_coeffs_idx[12] == 18);
	assert(o_coeffs_idx[13] == 19);
	assert(o_coeffs_idx[14] == 20);
	assert(o_coeffs_idx[15] == 21);
	assert(o_coeffs_idx[16] == 22);
	assert(o_coeffs_idx[17] == 23);
	assert(o_coeffs_idx[18] == 54+28);
	assert(o_coeffs_idx[19] == 54+29);
	assert(o_coeffs_idx[20] == 54+31);
	assert(o_coeffs_idx[21] == 54+32);
	assert(o_coeffs_idx[22] == 54+34);
	assert(o_coeffs_idx[23] == 54+35);
	assert(o_coeffs_idx[24] == 54+36);
	assert(o_coeffs_idx[25] == 54+37);
	assert(o_coeffs_idx[26] == 54+39);
	assert(o_coeffs_idx[27] == 54+40);
	assert(o_coeffs_idx[28] == 54+42);
	assert(o_coeffs_idx[29] == 54+43);
	assert(o_coeffs_idx[30] == 54+45);
	assert(o_coeffs_idx[31] == 54+48);
	assert(o_coeffs_idx[32] == 54+51);
	/* Last stored patch. */
	assert(((double*)coding_tmps)[0] == 1);
	assert(((double*)coding_tmps)[1] == 1);
	assert(((double*)coding_tmps)[2] == 0);
	assert(((double*)coding_tmps)[3] == 0);
	assert(((double*)coding_tmps)[4] == 0);
	assert(((double*)coding_tmps)[5] == 0);
	assert(((double*)coding_tmps)[6] == 0);
	assert(((double*)coding_tmps)[7] == 0);
	assert(((double*)coding_tmps)[8] == 0);
	/* Coded patches. */
	assert(((double*)coding_tmps)[9+0] == 1);
	assert(((double*)coding_tmps)[9+1] == -0.5);
	assert(((double*)coding_tmps)[9+2] == 1);
	assert(((double*)coding_tmps)[9+3] == -0.5);
	assert(((double*)coding_tmps)[9+4] == 0.5);
	assert(((double*)coding_tmps)[9+5] == -1);
	assert(((double*)coding_tmps)[9+6] == 1);
	assert(((double*)coding_tmps)[9+7] == -0.5);
	assert(((double*)coding_tmps)[9+8] == 1);
	assert(((double*)coding_tmps)[9+9] == -0.5);
	assert(((double*)coding_tmps)[9+10] == 1);
	assert(((double*)coding_tmps)[9+11] == -0.5);
	assert(((double*)coding_tmps)[9+12] == 1);
	assert(((double*)coding_tmps)[9+13] == 0.5);
	assert(((double*)coding_tmps)[9+14] == 1);
	assert(((double*)coding_tmps)[9+15] == -0.5);
	assert(((double*)coding_tmps)[9+16] == 1);
	assert(((double*)coding_tmps)[9+17] == -0.5);
	assert(((double*)coding_tmps)[9+18] == 1);
	assert(((double*)coding_tmps)[9+19] == -0.5);
	assert(((double*)coding_tmps)[9+20] == 1);
	assert(((double*)coding_tmps)[9+21] == -0.5);
	assert(((double*)coding_tmps)[9+22] == 1);
	assert(((double*)coding_tmps)[9+23] == -0.5);
	assert(((double*)coding_tmps)[9+24] == 0.5);
	assert(((double*)coding_tmps)[9+25] == -1);
	assert(((double*)coding_tmps)[9+26] == 0.5);
	assert(((double*)coding_tmps)[9+27] == -1);
	assert(((double*)coding_tmps)[9+28] == 0.5);
	assert(((double*)coding_tmps)[9+29] == -1);
	assert(((double*)coding_tmps)[9+30] == 0.5);
	assert(((double*)coding_tmps)[9+31] == -1);
	assert(((double*)coding_tmps)[9+32] == 1);
	assert(((double*)coding_tmps)[9+33] == -0.5);
	assert(((double*)coding_tmps)[9+34] == 1);
	assert(((double*)coding_tmps)[9+35] == -0.5);
	assert(((double*)coding_tmps)[9+36] == 1);
	assert(((double*)coding_tmps)[9+37] == -0.5);
	assert(((double*)coding_tmps)[9+38] == 1);
	assert(((double*)coding_tmps)[9+39] == -0.5);
	assert(((double*)coding_tmps)[9+40] == 1);
	assert(((double*)coding_tmps)[9+41] == -0.5);
	assert(((double*)coding_tmps)[9+42] == 1);
	assert(((double*)coding_tmps)[9+43] == -0.5);
	assert(((double*)coding_tmps)[9+44] == 1);
	assert(((double*)coding_tmps)[9+45] == -0.5);
	assert(((double*)coding_tmps)[9+46] == 1);
	assert(((double*)coding_tmps)[9+47] == -0.5);
	assert(((double*)coding_tmps)[9+48] == 0.5);
	assert(((double*)coding_tmps)[9+49] == -1);
	assert(((double*)coding_tmps)[9+50] == 1);
	assert(((double*)coding_tmps)[9+51] == -0.5);
	assert(((double*)coding_tmps)[9+52] == 1);
	assert(((double*)coding_tmps)[9+53] == -0.5);
	assert(((double*)coding_tmps)[9+54] == 1);
	assert(((double*)coding_tmps)[9+55] == -0.5);
	assert(((double*)coding_tmps)[9+56] == 0.5);
	assert(((double*)coding_tmps)[9+57] == 1);
	assert(((double*)coding_tmps)[9+58] == 1);
	assert(((double*)coding_tmps)[9+59] == -0.5);
	assert(((double*)coding_tmps)[9+60] == 1);
	assert(((double*)coding_tmps)[9+61] == -0.5);
	assert(((double*)coding_tmps)[9+62] == 1);
	assert(((double*)coding_tmps)[9+63] == -0.5);
	assert(((double*)coding_tmps)[9+64] == 1);
	assert(((double*)coding_tmps)[9+65] == -0.5);
	assert(((double*)coding_tmps)[9+66] == 1);
	assert(((double*)coding_tmps)[9+67] == -0.5);
       	assert(((double*)coding_tmps)[9+68] == 1);
	assert(((double*)coding_tmps)[9+69] == -0.5);
	assert(((double*)coding_tmps)[9+70] == 1);
	assert(((double*)coding_tmps)[9+71] == -0.5);
	/* Coded idx patches.*/
	assert(((size_t*)(((double*)coding_tmps)+9+72))[0] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[1] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[2] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[3] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[4] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[5] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[6] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[7] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[8] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[9] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[10] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[11] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[12] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[13] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[14] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[15] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[16] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[17] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[18] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[19] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[20] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[21] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[22] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[23] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[24] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[25] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[26] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[27] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[28] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[29] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[30] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[31] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[32] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[33] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[34] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[35] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[36] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[37] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[38] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[39] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[40] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[41] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[42] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[43] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[44] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[45] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[46] == 2);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[47] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[48] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[49] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[50] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[51] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[52] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[53] == 11);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[54] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[55] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[56] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[57] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[58] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[59] == 10);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[60] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[61] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[62] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[63] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[64] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[65] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[66] == 1);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[67] == 9);
       	assert(((size_t*)(((double*)coding_tmps)+9+72))[68] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[69] == 9);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[70] == 0);
	assert(((size_t*)(((double*)coding_tmps)+9+72))[71] == 9);
	/* Temporary coding data. */
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[0] == 2);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[1] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[2] == -1);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[3] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[4] == 0);
	assert(((double*)((size_t*)((double*)coding_tmps+9+72)+72))[5] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[0] == 0);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[1] == 3);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[2] == 4);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[3] == 1);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[4] == 2);
	assert(((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6))[5] == 5);
	/* Reduce data. */
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[0] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[1] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[2] == 2);
	assert(((size_t*)((size_t*)((double*)((size_t*)((double*)coding_tmps+9+72)+72)+6)+6))[3] == 2);

	free(o_coeffs);
	free(o_coeffs_idx);
        free(coding_tmps);
    }

