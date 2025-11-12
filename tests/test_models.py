"""
Tests for model training and evaluation functions.
"""
import unittest
import numpy as np
import tempfile
import os
from models.model import (
    save_trained_model,
    cosine_annealing_schedule,
    residual_block,
    transformer_block,
    attention_lstm_layer,
    get_channel_config,
    train_hybrid_model,
    evaluate_model
)
import tensorflow as tf

class TestModelFunctions(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_save_trained_model(self):
        """Test model saving functionality"""
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5, 1)),
            tf.keras.layers.Conv1D(32, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        # Save model
        model_path = os.path.join(self.temp_dir, "test_model.h5")
        save_trained_model(model, model_path)
        
        # Verify model was saved
        self.assertTrue(os.path.exists(model_path))
        
        # Load and verify
        loaded_model = tf.keras.models.load_model(model_path)
        self.assertEqual(len(model.layers), len(loaded_model.layers))
    
    def test_cosine_annealing_schedule(self):
        """Test cosine annealing learning rate schedule"""
        initial_lr = 0.001
        
        # Test at different epochs
        lr_epoch_0 = cosine_annealing_schedule(0, initial_lr)
        lr_epoch_50 = cosine_annealing_schedule(50, initial_lr)
        lr_epoch_100 = cosine_annealing_schedule(100, initial_lr)
        
        # Learning rate should decrease over time
        self.assertGreater(lr_epoch_0, lr_epoch_50)
        self.assertGreater(lr_epoch_50, lr_epoch_100)
        
        # Learning rate should be positive
        self.assertGreater(lr_epoch_0, 0)
        self.assertGreater(lr_epoch_100, 0)
    
    def test_residual_block(self):
        """Test residual block construction"""
        # Create input tensor
        inputs = tf.keras.layers.Input(shape=(10, 32))
        
        # Apply residual block
        output = residual_block(inputs, filters=64, kernel_size=3, dropout_rate=0.3)
        
        # Verify output shape
        self.assertEqual(output.shape[-1], 64)
    
    def test_transformer_block(self):
        """Test transformer block construction"""
        # Create input tensor
        inputs = tf.keras.layers
est.main()itt:
    unmain__'me__ == '__
if __nacy'], 1)
ics['accurassEqual(metrtLeself.asser
        ], 0)acy'rics['accural(meterEquassertGreatlf.
        send 1 0 a is betweenccuracy aerify # V         
  
    , metrics)report'ification_('classassertIn     self.cs)
    metritrix',_maionconfussertIn('.as self     )
  cy', metricsuraertIn('accass self.       d
ulate were calcmetricsify     # Ver    
        
ate=False)test, calibrst, y_, X_te(modelluate_model= evametrics     l
    aluate mode     # Ev     
   30)
   , 3, (0dom.randintp.ranst = n        y_tedn(30, 5)
andom.ran_test = np.r  X
       test dataerate dummy Gen     #       
   cy'])
 uras=['acc, metricentropy'al_crossse_categoric loss='spardam',zer='aptimil.compile(ode  mo
      
        ])='softmax')tion activae(3,ensas.layers.D.ker          tften(),
  yers.Flatras.la       tf.ke
     ion='relu'),3, activatv1D(32, s.Cons.layer     tf.kera     )),
   1(shape=(5,.Inputas.layersker  tf.  
        ntial([s.Sequera tf.keodel =
        msimple modelate a       # Cre
  "ion""luateva"Test model ""       ):
 selfte_model(aluat_evef tes 
    d   sses
3)  # 3 clashape[1], ions.qual(predictassertE     self., 5)
   .shape[0]redictionsertEqual(p  self.ass   )
   e(-1, 5, 1):5].reshapn[ct(X_trai model.prediictions =    pred   ons
  predictidel can makey mo     # Verif      
 ry)
    toisIsNotNone(hsertself.asl)
        tNone(modeassertIsNo self.
       createdodel was ify m      # Ver 
            )
=16
     size batch_   
        s=2,    epoch,
        al'type='origin  model_         
  y_train,rain, X_t           odel(
brid_m= train_hy, history odel     mpochs
    einimalodel with m# Train m          
0)
       5t(0, 3,dindom.ranain = np.ran   y_tr5)
     andn(50, ndom.rrain = np.ra_t       X
 g datay traininrate dumm      # Gene  
ning"""model traist hybrid ""Te       "
 f):el(seld_modhybrirain_ def test_t 
   '])
   nitstm_uonfig_64['lsits'], c_8['lstm_uns(configsertLeslf.as     se)
   arger modelsave lnts should hl couger channealing (lar sc# Verify  
        
      ), config_heads'ttentionssertIn('aself.a       
     g)nits', confi'dense_uassertIn(     self.
       onfig), cs'tm_unittIn('lssser self.a       nfig)
    ilters', cotIn('conv_fer self.ass          nfig_64]:
  co_32,_16, config_8, confignfigin [co config  for       ired keys
requns have configuratio   # Verify 
     
        4)_config(6elget_channconfig_64 = 2)
        ig(3nfnel_cohan_c getg_32 =     confi
   _config(16)annelet_ch_16 = gnfig     config(8)
   el_co_channet_8 = g      configtions
  configurasupported Test         # l"""
trievaion reat configurnnel""Test cha      "):
  selfnfig(el_coet_chann def test_g)
    
   , 2hape)utput.sn(oual(lessertEqf.a    sel    ooling)
n pentioter att(aft is 1D y outpu  # Verif 
          
   _rate=0.3)4, dropoutits=6uts, unm_layer(inplstttention_ = a    output
    STM layerion Lattent  # Apply 
      )
        32)10, ut(shape=(npyers.Ikeras.las = tf.  inputsor
       tenputinCreate         # n"""
structioM layer conSTention L""Test att       "
 f):(seltm_layertion_lst_atten    def tes 
1], 64)
   ape[-tput.shqual(ouelf.assertE  s      put
hes inshape matcoutput rify     # Ve
                  )
 0.1
    dropout=         128, 
     ff_dim=
       m_heads=4,   nu      , 
    mbed_dim=64          es, 
   input          
 er_block(transformtput =        our block
  transforme # Apply
          
     ))e=(10, 64.Input(shap