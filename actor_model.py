import torch
import numpy as np
import transformer_engine.pytorch as te
import os
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
import torch.jit as jit

time_size = 256

class ActorModel(torch.nn.Module):
    def __init__(self, batch_size = 64, transformer_size=1024, transformer_attention_size=64, dropout=0.1, fuse_qkv=False):
        super().__init__()
        self.batch_size = batch_size
        self.state_embedding = torch.nn.Linear(208, transformer_size)
        
        self.encoder_transformer_layer_1 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size, 
                                                     num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=fuse_qkv,
                                                     
                                                     )
        self.encoder_transformer_layer_2 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size, 
                                                     num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=fuse_qkv, 
                                                     
                                                     )
        self.encoder_transformer_layer_3 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size, 
                                                     num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=fuse_qkv, 
                                                     
                                                     )
        self.encoder_transformer_layer_4 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size, 
                                                     num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=fuse_qkv, 
                                                     
                                                     )
        self.encoder_transformer_layer_5 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=fuse_qkv,
                                                        
                                                        )
        self.encoder_transformer_layer_6 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=fuse_qkv,
                                                        
                                                        )
        self.encoder_transformer_layer_7 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=fuse_qkv,
                                                        
                                                        )
        self.encoder_transformer_layer_8 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=fuse_qkv,
                                                        
                                                        )
        self.encoder_transformer_layer_9 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=fuse_qkv,
                                                        
                                                        )
        self.encoder_transformer_layer_10 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=fuse_qkv,
                                                        
                                                        )
        self.encoder_transformer_layer_11 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=fuse_qkv,
                                                        
                                                        )
        self.encoder_transformer_layer_12 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='encoder', fuse_qkv_params=fuse_qkv,
                                                        
                                                        )
        

        self.decoder_transformer_layer_1 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size, 
                                                     num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=fuse_qkv,
                                                     
                                                     )
        self.decoder_transformer_layer_2 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=fuse_qkv,
                                                        
                                                        )
        self.decoder_transformer_layer_3 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=fuse_qkv,
                                                        
                                                        )
        self.decoder_transformer_layer_4 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=fuse_qkv,
                                                        
                                                        )
        self.decoder_transformer_layer_5 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=fuse_qkv,
                                                        
                                                        )
        self.decoder_transformer_layer_6 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=fuse_qkv,
                                                        
                                                        )
        self.decoder_transformer_layer_7 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=fuse_qkv,
                                                        
                                                        )
        self.decoder_transformer_layer_8 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=fuse_qkv,
                                                        
                                                        )
        self.decoder_transformer_layer_9 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                        num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=fuse_qkv,
                                                        
                                                        )
        self.decoder_transformer_layer_10 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                            num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=fuse_qkv,
                                                            
                                                            )
        self.decoder_transformer_layer_11 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                            num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=fuse_qkv,
                                                            
                                                            )
        self.decoder_transformer_layer_12 = te.TransformerLayer(hidden_size=transformer_size, ffn_hidden_size=transformer_size,
                                                            num_attention_heads=transformer_attention_size, layer_type='decoder', fuse_qkv_params=fuse_qkv,
                                                            
                                                            )
        
        self.state_layer = torch.nn.Linear(3, 512)
        self.transformer_output_layer = te.Linear(transformer_size, 512)

        self.actor_layer_1 = te.Linear(transformer_size, 256)
        self.actor_layer_2 = te.Linear(256, 128)
        self.actor_layer_3 = torch.nn.Linear(128*time_size, 11)
                
        self.critic_layer_1 = te.Linear(128, 1)

        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        #self.final_linear = te.Linear(transformer_size, 208, tp_group=tensor_parallel_group, sequence_parallel=True)

        self.positional_encoder = Summer(PositionalEncoding1D(transformer_size)).cuda()

        self.padding = torch.zeros((self.batch_size, time_size, 4, 2)).cuda()

    def forward(self, mask, input_tuple, env_state):
        padded = torch.cat((input_tuple, self.padding), axis=1)
        padded_mask = torch.cat((mask, self.padding), axis=1)

        input_tuple = self.state_embedding(padded)
        
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
        state_output = self.relu(state_output)
        state_output = self.dropout(state_output)

        transformer_output = self.transformer_output_layer(decoder_output)
        transformer_output = self.relu(transformer_output)
        transformer_output = self.dropout(transformer_output)

        full_state = torch.cat((state_output, transformer_output), axis=-1)
        
        full_state = self.actor_layer_1(full_state)
        full_state = self.relu(full_state)
        full_state = self.dropout(full_state)
        
        full_state = self.actor_layer_2(full_state)

        full_state = torch.reshape(full_state, (self.batch_size, -1))

        full_state = self.relu(full_state)
        full_state = self.dropout(full_state)
        
        critic_output = self.critic_layer_1(full_state)
        critic_output = self.relu(critic_output)

        actor_output = self.actor_layer_3(full_state)
        actor_output = self.relu(actor_output)
        actor_output = torch.nn.functional.softmax(actor_output, dim=-2)
        
        return actor_output, critic_output
