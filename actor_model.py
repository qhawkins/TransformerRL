import torch
import numpy as np
import transformer_engine.pytorch as te
import os
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
import torch.jit as jit

time_size = 256

class ActorModel(torch.nn.Module):
    def __init__(self, rank, tensor_parallel_group, transformer_size=1024, transformer_attention_size=32, batch_size=512, dropout=0.0):
        super().__init__()
        self.batch_size = batch_size
        self.state_embedding = te.Linear(208, transformer_size, tp_group=tensor_parallel_group, sequence_parallel=True)
        self.rank = rank
        self.encoder_transformer_layer_1 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size, 
                                                     num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=True,
                                                     set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                     )
        self.encoder_transformer_layer_2 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size, 
                                                     num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=True,
                                                     set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                     )
        self.encoder_transformer_layer_3 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size, 
                                                     num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=True,
                                                     set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                     )
        self.encoder_transformer_layer_4 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size, 
                                                     num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=True,
                                                     set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                     )
        self.encoder_transformer_layer_5 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=True,
                                                        set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                        )
        self.encoder_transformer_layer_6 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=True,
                                                        set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                        )
        self.encoder_transformer_layer_7 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=True,
                                                        set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                        )
        self.encoder_transformer_layer_8 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=True,
                                                        set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                        )
        self.encoder_transformer_layer_9 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=True,
                                                        set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                        )
        self.encoder_transformer_layer_10 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=True,
                                                        set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                        )
        self.encoder_transformer_layer_11 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=True,
                                                        set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                        )
        self.encoder_transformer_layer_12 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=True,
                                                        set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                        )
        

        self.decoder_transformer_layer_1 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size, 
                                                     num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=True,
                                                     set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                     )
        self.decoder_transformer_layer_2 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=True,
                                                        set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                        )
        self.decoder_transformer_layer_3 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=True,
                                                        set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                        )
        self.decoder_transformer_layer_4 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=True,
                                                        set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                        )
        self.decoder_transformer_layer_5 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=True,
                                                        set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                        )
        self.decoder_transformer_layer_6 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=True,
                                                        set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                        )
        self.decoder_transformer_layer_7 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=True,
                                                        set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                        )
        self.decoder_transformer_layer_8 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=True,
                                                        set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                        )
        self.decoder_transformer_layer_9 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=True,
                                                        set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                        )
        self.decoder_transformer_layer_10 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                            num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=True,
                                                            set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                            )
        self.decoder_transformer_layer_11 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                            num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=True,
                                                            set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                            )
        self.decoder_transformer_layer_12 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                            num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=True,
                                                            set_parallel_mode=True, tp_group=tensor_parallel_group, sequence_parallel=True
                                                            )
        
        self.state_layer = torch.nn.Linear(3, 512)
        self.transformer_output_layer = torch.nn.Linear(transformer_size, 512)

        self.actor_layer_1 = torch.nn.Linear(1024, 256)
        self.actor_layer_2 = torch.nn.Linear(256, 128)
        self.actor_layer_3 = torch.nn.Linear(128*256, 11)
                
        self.dropout = torch.nn.Dropout(dropout)

        #self.final_linear = te.Linear(transformer_size, 208, tp_group=tensor_parallel_group, sequence_parallel=True)

        self.positional_encoder = Summer(PositionalEncoding1D(transformer_size))

        self.padding = torch.zeros((self.batch_size, time_size, 4, 2)).cuda()

    def forward(self, mask, input_tuple, env_state):
        padded = torch.cat((input_tuple, self.padding), axis=2)
        padded_mask = torch.cat((mask, self.padding), axis=2)
        input_tuple = self.state_embedding(torch.reshape(padded, (self.batch_size, time_size, 208)))
        input_tuple = self.positional_encoder(input_tuple)
        input_tuple = self.dropout(input_tuple)
        

        encoder_output = self.encoder_transformer_layer_1(input_tuple)
        encoder_output = self.encoder_transformer_layer_2(encoder_output)
        encoder_output = self.encoder_transformer_layer_3(encoder_output)
        encoder_output = self.encoder_transformer_layer_4(encoder_output)
        encoder_output = self.encoder_transformer_layer_5(encoder_output)
        encoder_output = self.encoder_transformer_layer_6(encoder_output)
        encoder_output = self.encoder_transformer_layer_7(encoder_output)
        encoder_output = self.encoder_transformer_layer_8(encoder_output)
        encoder_output = self.encoder_transformer_layer_9(encoder_output)
        encoder_output = self.encoder_transformer_layer_10(encoder_output)
        encoder_output = self.encoder_transformer_layer_11(encoder_output)
        encoder_output = self.encoder_transformer_layer_12(encoder_output)
        
        decoder_output = self.decoder_transformer_layer_1(attention_mask=padded_mask, hidden_states=input_tuple, encoder_output=encoder_output)
        decoder_output = self.decoder_transformer_layer_2(attention_mask=padded_mask, hidden_states=decoder_output, encoder_output=encoder_output)
        decoder_output = self.decoder_transformer_layer_3(attention_mask=padded_mask, hidden_states=decoder_output, encoder_output=encoder_output)
        decoder_output = self.decoder_transformer_layer_4(attention_mask=padded_mask, hidden_states=decoder_output, encoder_output=encoder_output)
        decoder_output = self.decoder_transformer_layer_5(attention_mask=padded_mask, hidden_states=decoder_output, encoder_output=encoder_output)
        decoder_output = self.decoder_transformer_layer_6(attention_mask=padded_mask, hidden_states=decoder_output, encoder_output=encoder_output)
        decoder_output = self.decoder_transformer_layer_7(attention_mask=padded_mask, hidden_states=decoder_output, encoder_output=encoder_output)
        decoder_output = self.decoder_transformer_layer_8(attention_mask=padded_mask, hidden_states=decoder_output, encoder_output=encoder_output)
        decoder_output = self.decoder_transformer_layer_9(attention_mask=padded_mask, hidden_states=decoder_output, encoder_output=encoder_output)
        decoder_output = self.decoder_transformer_layer_10(attention_mask=padded_mask, hidden_states=decoder_output, encoder_output=encoder_output)
        decoder_output = self.decoder_transformer_layer_11(attention_mask=padded_mask, hidden_states=decoder_output, encoder_output=encoder_output)
        decoder_output = self.decoder_transformer_layer_12(attention_mask=padded_mask, hidden_states=decoder_output, encoder_output=encoder_output)

        state_output = self.state_layer(env_state)
        state_output = self.dropout(state_output)

        transformer_output = self.transformer_output_layer(decoder_output)

        full_state = torch.cat((state_output, transformer_output), axis=-1)
        full_state = self.actor_layer_1(full_state)
        full_state = self.actor_layer_2(full_state)
        full_state = torch.flatten(full_state, start_dim=1)
        output = self.actor_layer_3(full_state)

        return output
