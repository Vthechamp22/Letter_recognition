´
Śý
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
ž
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.2.02v2.2.0-0-g2b96f3662b8Ě¤

conv2d_148/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_148/kernel

%conv2d_148/kernel/Read/ReadVariableOpReadVariableOpconv2d_148/kernel*'
_output_shapes
:*
dtype0
w
conv2d_148/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_148/bias
p
#conv2d_148/bias/Read/ReadVariableOpReadVariableOpconv2d_148/bias*
_output_shapes	
:*
dtype0

conv2d_149/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_149/kernel

%conv2d_149/kernel/Read/ReadVariableOpReadVariableOpconv2d_149/kernel*(
_output_shapes
:*
dtype0
w
conv2d_149/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_149/bias
p
#conv2d_149/bias/Read/ReadVariableOpReadVariableOpconv2d_149/bias*
_output_shapes	
:*
dtype0
~
dense_154/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_154/kernel
w
$dense_154/kernel/Read/ReadVariableOpReadVariableOpdense_154/kernel* 
_output_shapes
:
*
dtype0
u
dense_154/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_154/bias
n
"dense_154/bias/Read/ReadVariableOpReadVariableOpdense_154/bias*
_output_shapes	
:*
dtype0
}
dense_155/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*!
shared_namedense_155/kernel
v
$dense_155/kernel/Read/ReadVariableOpReadVariableOpdense_155/kernel*
_output_shapes
:	*
dtype0
t
dense_155/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_155/bias
m
"dense_155/bias/Read/ReadVariableOpReadVariableOpdense_155/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv2d_148/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_148/kernel/m

,Adam/conv2d_148/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_148/kernel/m*'
_output_shapes
:*
dtype0

Adam/conv2d_148/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_148/bias/m
~
*Adam/conv2d_148/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_148/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_149/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_149/kernel/m

,Adam/conv2d_149/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_149/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_149/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_149/bias/m
~
*Adam/conv2d_149/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_149/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_154/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_154/kernel/m

+Adam/dense_154/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_154/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_154/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_154/bias/m
|
)Adam/dense_154/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_154/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_155/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/dense_155/kernel/m

+Adam/dense_155/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_155/kernel/m*
_output_shapes
:	*
dtype0

Adam/dense_155/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_155/bias/m
{
)Adam/dense_155/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_155/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_148/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_148/kernel/v

,Adam/conv2d_148/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_148/kernel/v*'
_output_shapes
:*
dtype0

Adam/conv2d_148/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_148/bias/v
~
*Adam/conv2d_148/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_148/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_149/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_149/kernel/v

,Adam/conv2d_149/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_149/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_149/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_149/bias/v
~
*Adam/conv2d_149/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_149/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_154/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_154/kernel/v

+Adam/dense_154/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_154/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_154/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_154/bias/v
|
)Adam/dense_154/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_154/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_155/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/dense_155/kernel/v

+Adam/dense_155/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_155/kernel/v*
_output_shapes
:	*
dtype0

Adam/dense_155/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_155/bias/v
{
)Adam/dense_155/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_155/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ů4
valueĎ4BĚ4 BĹ4
´
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
		variables

regularization_losses
trainable_variables
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
 trainable_variables
!	keras_api
R
"	variables
#regularization_losses
$trainable_variables
%	keras_api
h

&kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
h

,kernel
-bias
.	variables
/regularization_losses
0trainable_variables
1	keras_api
Đ
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmkmlmm&mn'mo,mp-mqvrvsvtvu&vv'vw,vx-vy
8
0
1
2
3
&4
'5
,6
-7
 
8
0
1
2
3
&4
'5
,6
-7
­
7layer_regularization_losses
		variables
8non_trainable_variables

regularization_losses
9layer_metrics
:metrics
trainable_variables

;layers
 
][
VARIABLE_VALUEconv2d_148/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_148/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
<layer_regularization_losses
	variables
=non_trainable_variables
regularization_losses
>layer_metrics
?metrics
trainable_variables

@layers
 
 
 
­
Alayer_regularization_losses
	variables
Bnon_trainable_variables
regularization_losses
Clayer_metrics
Dmetrics
trainable_variables

Elayers
][
VARIABLE_VALUEconv2d_149/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_149/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
Flayer_regularization_losses
	variables
Gnon_trainable_variables
regularization_losses
Hlayer_metrics
Imetrics
trainable_variables

Jlayers
 
 
 
­
Klayer_regularization_losses
	variables
Lnon_trainable_variables
regularization_losses
Mlayer_metrics
Nmetrics
 trainable_variables

Olayers
 
 
 
­
Player_regularization_losses
"	variables
Qnon_trainable_variables
#regularization_losses
Rlayer_metrics
Smetrics
$trainable_variables

Tlayers
\Z
VARIABLE_VALUEdense_154/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_154/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1
 

&0
'1
­
Ulayer_regularization_losses
(	variables
Vnon_trainable_variables
)regularization_losses
Wlayer_metrics
Xmetrics
*trainable_variables

Ylayers
\Z
VARIABLE_VALUEdense_155/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_155/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1
 

,0
-1
­
Zlayer_regularization_losses
.	variables
[non_trainable_variables
/regularization_losses
\layer_metrics
]metrics
0trainable_variables

^layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

_0
`1
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	atotal
	bcount
c	variables
d	keras_api
D
	etotal
	fcount
g
_fn_kwargs
h	variables
i	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

a0
b1

c	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

e0
f1

h	variables
~
VARIABLE_VALUEAdam/conv2d_148/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_148/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_149/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_149/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_154/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_154/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_155/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_155/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_148/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_148/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_149/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_149/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_154/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_154/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_155/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_155/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

 serving_default_conv2d_148_inputPlaceholder*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*$
shape:˙˙˙˙˙˙˙˙˙
ś
StatefulPartitionedCallStatefulPartitionedCall serving_default_conv2d_148_inputconv2d_148/kernelconv2d_148/biasconv2d_149/kernelconv2d_149/biasdense_154/kerneldense_154/biasdense_155/kerneldense_155/bias*
Tin
2	*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*.
f)R'
%__inference_signature_wrapper_5497050
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¨
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_148/kernel/Read/ReadVariableOp#conv2d_148/bias/Read/ReadVariableOp%conv2d_149/kernel/Read/ReadVariableOp#conv2d_149/bias/Read/ReadVariableOp$dense_154/kernel/Read/ReadVariableOp"dense_154/bias/Read/ReadVariableOp$dense_155/kernel/Read/ReadVariableOp"dense_155/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/conv2d_148/kernel/m/Read/ReadVariableOp*Adam/conv2d_148/bias/m/Read/ReadVariableOp,Adam/conv2d_149/kernel/m/Read/ReadVariableOp*Adam/conv2d_149/bias/m/Read/ReadVariableOp+Adam/dense_154/kernel/m/Read/ReadVariableOp)Adam/dense_154/bias/m/Read/ReadVariableOp+Adam/dense_155/kernel/m/Read/ReadVariableOp)Adam/dense_155/bias/m/Read/ReadVariableOp,Adam/conv2d_148/kernel/v/Read/ReadVariableOp*Adam/conv2d_148/bias/v/Read/ReadVariableOp,Adam/conv2d_149/kernel/v/Read/ReadVariableOp*Adam/conv2d_149/bias/v/Read/ReadVariableOp+Adam/dense_154/kernel/v/Read/ReadVariableOp)Adam/dense_154/bias/v/Read/ReadVariableOp+Adam/dense_155/kernel/v/Read/ReadVariableOp)Adam/dense_155/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__traced_save_5497341

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_148/kernelconv2d_148/biasconv2d_149/kernelconv2d_149/biasdense_154/kerneldense_154/biasdense_155/kerneldense_155/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_148/kernel/mAdam/conv2d_148/bias/mAdam/conv2d_149/kernel/mAdam/conv2d_149/bias/mAdam/dense_154/kernel/mAdam/dense_154/bias/mAdam/dense_155/kernel/mAdam/dense_155/bias/mAdam/conv2d_148/kernel/vAdam/conv2d_148/bias/vAdam/conv2d_149/kernel/vAdam/conv2d_149/bias/vAdam/dense_154/kernel/vAdam/dense_154/bias/vAdam/dense_155/kernel/vAdam/dense_155/bias/v*-
Tin&
$2"*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference__traced_restore_5497452Â


Ţ
/__inference_sequential_79_layer_call_fn_5497143

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity˘StatefulPartitionedCallŚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_sequential_79_layer_call_and_return_conditional_losses_54969522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:˙˙˙˙˙˙˙˙˙::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Á
c
G__inference_flatten_79_layer_call_and_return_conditional_losses_5497170

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ň
Ž
F__inference_dense_155_layer_call_and_return_conditional_losses_5497206

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ż

Ż
G__inference_conv2d_149_layer_call_and_return_conditional_losses_5496788

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpˇ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

j
N__inference_max_pooling2d_148_layer_call_and_return_conditional_losses_5496770

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
!
Á
J__inference_sequential_79_layer_call_and_return_conditional_losses_5496922
conv2d_148_input
conv2d_148_5496898
conv2d_148_5496900
conv2d_149_5496904
conv2d_149_5496906
dense_154_5496911
dense_154_5496913
dense_155_5496916
dense_155_5496918
identity˘"conv2d_148/StatefulPartitionedCall˘"conv2d_149/StatefulPartitionedCall˘!dense_154/StatefulPartitionedCall˘!dense_155/StatefulPartitionedCall
"conv2d_148/StatefulPartitionedCallStatefulPartitionedCallconv2d_148_inputconv2d_148_5496898conv2d_148_5496900*
Tin
2*
Tout
2*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_148_layer_call_and_return_conditional_losses_54967542$
"conv2d_148/StatefulPartitionedCallü
!max_pooling2d_148/PartitionedCallPartitionedCall+conv2d_148/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_max_pooling2d_148_layer_call_and_return_conditional_losses_54967702#
!max_pooling2d_148/PartitionedCallŹ
"conv2d_149/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_148/PartitionedCall:output:0conv2d_149_5496904conv2d_149_5496906*
Tin
2*
Tout
2*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_149_layer_call_and_return_conditional_losses_54967882$
"conv2d_149/StatefulPartitionedCallü
!max_pooling2d_149/PartitionedCallPartitionedCall+conv2d_149/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_max_pooling2d_149_layer_call_and_return_conditional_losses_54968042#
!max_pooling2d_149/PartitionedCallŢ
flatten_79/PartitionedCallPartitionedCall*max_pooling2d_149/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_flatten_79_layer_call_and_return_conditional_losses_54968322
flatten_79/PartitionedCall
!dense_154/StatefulPartitionedCallStatefulPartitionedCall#flatten_79/PartitionedCall:output:0dense_154_5496911dense_154_5496913*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_154_layer_call_and_return_conditional_losses_54968512#
!dense_154/StatefulPartitionedCall
!dense_155/StatefulPartitionedCallStatefulPartitionedCall*dense_154/StatefulPartitionedCall:output:0dense_155_5496916dense_155_5496918*
Tin
2*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_155_layer_call_and_return_conditional_losses_54968782#
!dense_155/StatefulPartitionedCall
IdentityIdentity*dense_155/StatefulPartitionedCall:output:0#^conv2d_148/StatefulPartitionedCall#^conv2d_149/StatefulPartitionedCall"^dense_154/StatefulPartitionedCall"^dense_155/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:˙˙˙˙˙˙˙˙˙::::::::2H
"conv2d_148/StatefulPartitionedCall"conv2d_148/StatefulPartitionedCall2H
"conv2d_149/StatefulPartitionedCall"conv2d_149/StatefulPartitionedCall2F
!dense_154/StatefulPartitionedCall!dense_154/StatefulPartitionedCall2F
!dense_155/StatefulPartitionedCall!dense_155/StatefulPartitionedCall:a ]
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameconv2d_148_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ľ

č
/__inference_sequential_79_layer_call_fn_5497019
conv2d_148_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity˘StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallconv2d_148_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_sequential_79_layer_call_and_return_conditional_losses_54970002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:˙˙˙˙˙˙˙˙˙::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameconv2d_148_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ź

Ż
G__inference_conv2d_148_layer_call_and_return_conditional_losses_5496754

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpˇ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
Relu
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:::i e
A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ó	
Ţ
%__inference_signature_wrapper_5497050
conv2d_148_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv2d_148_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference__wrapped_model_54967422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:˙˙˙˙˙˙˙˙˙::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameconv2d_148_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ý

+__inference_dense_155_layer_call_fn_5497215

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_155_layer_call_and_return_conditional_losses_54968782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ç

,__inference_conv2d_148_layer_call_fn_5496764

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallđ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_148_layer_call_and_return_conditional_losses_54967542
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
đ
Ž
F__inference_dense_154_layer_call_and_return_conditional_losses_5497186

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
!
Á
J__inference_sequential_79_layer_call_and_return_conditional_losses_5496895
conv2d_148_input
conv2d_148_5496814
conv2d_148_5496816
conv2d_149_5496820
conv2d_149_5496822
dense_154_5496862
dense_154_5496864
dense_155_5496889
dense_155_5496891
identity˘"conv2d_148/StatefulPartitionedCall˘"conv2d_149/StatefulPartitionedCall˘!dense_154/StatefulPartitionedCall˘!dense_155/StatefulPartitionedCall
"conv2d_148/StatefulPartitionedCallStatefulPartitionedCallconv2d_148_inputconv2d_148_5496814conv2d_148_5496816*
Tin
2*
Tout
2*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_148_layer_call_and_return_conditional_losses_54967542$
"conv2d_148/StatefulPartitionedCallü
!max_pooling2d_148/PartitionedCallPartitionedCall+conv2d_148/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_max_pooling2d_148_layer_call_and_return_conditional_losses_54967702#
!max_pooling2d_148/PartitionedCallŹ
"conv2d_149/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_148/PartitionedCall:output:0conv2d_149_5496820conv2d_149_5496822*
Tin
2*
Tout
2*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_149_layer_call_and_return_conditional_losses_54967882$
"conv2d_149/StatefulPartitionedCallü
!max_pooling2d_149/PartitionedCallPartitionedCall+conv2d_149/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_max_pooling2d_149_layer_call_and_return_conditional_losses_54968042#
!max_pooling2d_149/PartitionedCallŢ
flatten_79/PartitionedCallPartitionedCall*max_pooling2d_149/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_flatten_79_layer_call_and_return_conditional_losses_54968322
flatten_79/PartitionedCall
!dense_154/StatefulPartitionedCallStatefulPartitionedCall#flatten_79/PartitionedCall:output:0dense_154_5496862dense_154_5496864*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_154_layer_call_and_return_conditional_losses_54968512#
!dense_154/StatefulPartitionedCall
!dense_155/StatefulPartitionedCallStatefulPartitionedCall*dense_154/StatefulPartitionedCall:output:0dense_155_5496889dense_155_5496891*
Tin
2*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_155_layer_call_and_return_conditional_losses_54968782#
!dense_155/StatefulPartitionedCall
IdentityIdentity*dense_155/StatefulPartitionedCall:output:0#^conv2d_148/StatefulPartitionedCall#^conv2d_149/StatefulPartitionedCall"^dense_154/StatefulPartitionedCall"^dense_155/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:˙˙˙˙˙˙˙˙˙::::::::2H
"conv2d_148/StatefulPartitionedCall"conv2d_148/StatefulPartitionedCall2H
"conv2d_149/StatefulPartitionedCall"conv2d_149/StatefulPartitionedCall2F
!dense_154/StatefulPartitionedCall!dense_154/StatefulPartitionedCall2F
!dense_155/StatefulPartitionedCall!dense_155/StatefulPartitionedCall:a ]
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameconv2d_148_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

O
3__inference_max_pooling2d_148_layer_call_fn_5496776

inputs
identityÍ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_max_pooling2d_148_layer_call_and_return_conditional_losses_54967702
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ž(
á
J__inference_sequential_79_layer_call_and_return_conditional_losses_5497122

inputs-
)conv2d_148_conv2d_readvariableop_resource.
*conv2d_148_biasadd_readvariableop_resource-
)conv2d_149_conv2d_readvariableop_resource.
*conv2d_149_biasadd_readvariableop_resource,
(dense_154_matmul_readvariableop_resource-
)dense_154_biasadd_readvariableop_resource,
(dense_155_matmul_readvariableop_resource-
)dense_155_biasadd_readvariableop_resource
identityˇ
 conv2d_148/Conv2D/ReadVariableOpReadVariableOp)conv2d_148_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02"
 conv2d_148/Conv2D/ReadVariableOpĆ
conv2d_148/Conv2DConv2Dinputs(conv2d_148/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
conv2d_148/Conv2DŽ
!conv2d_148/BiasAdd/ReadVariableOpReadVariableOp*conv2d_148_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!conv2d_148/BiasAdd/ReadVariableOpľ
conv2d_148/BiasAddBiasAddconv2d_148/Conv2D:output:0)conv2d_148/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_148/BiasAdd
conv2d_148/ReluReluconv2d_148/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_148/ReluÎ
max_pooling2d_148/MaxPoolMaxPoolconv2d_148/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2
max_pooling2d_148/MaxPool¸
 conv2d_149/Conv2D/ReadVariableOpReadVariableOp)conv2d_149_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02"
 conv2d_149/Conv2D/ReadVariableOpâ
conv2d_149/Conv2DConv2D"max_pooling2d_148/MaxPool:output:0(conv2d_149/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
conv2d_149/Conv2DŽ
!conv2d_149/BiasAdd/ReadVariableOpReadVariableOp*conv2d_149_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!conv2d_149/BiasAdd/ReadVariableOpľ
conv2d_149/BiasAddBiasAddconv2d_149/Conv2D:output:0)conv2d_149/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_149/BiasAdd
conv2d_149/ReluReluconv2d_149/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_149/ReluÎ
max_pooling2d_149/MaxPoolMaxPoolconv2d_149/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2
max_pooling2d_149/MaxPoolu
flatten_79/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  2
flatten_79/ConstĽ
flatten_79/ReshapeReshape"max_pooling2d_149/MaxPool:output:0flatten_79/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
flatten_79/Reshape­
dense_154/MatMul/ReadVariableOpReadVariableOp(dense_154_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_154/MatMul/ReadVariableOp§
dense_154/MatMulMatMulflatten_79/Reshape:output:0'dense_154/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_154/MatMulŤ
 dense_154/BiasAdd/ReadVariableOpReadVariableOp)dense_154_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_154/BiasAdd/ReadVariableOpŞ
dense_154/BiasAddBiasAdddense_154/MatMul:product:0(dense_154/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_154/BiasAddw
dense_154/ReluReludense_154/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_154/ReluŹ
dense_155/MatMul/ReadVariableOpReadVariableOp(dense_155_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02!
dense_155/MatMul/ReadVariableOp§
dense_155/MatMulMatMuldense_154/Relu:activations:0'dense_155/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_155/MatMulŞ
 dense_155/BiasAdd/ReadVariableOpReadVariableOp)dense_155_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_155/BiasAdd/ReadVariableOpŠ
dense_155/BiasAddBiasAdddense_155/MatMul:product:0(dense_155/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_155/BiasAdd
dense_155/SoftmaxSoftmaxdense_155/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_155/Softmaxo
IdentityIdentitydense_155/Softmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:˙˙˙˙˙˙˙˙˙:::::::::W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

j
N__inference_max_pooling2d_149_layer_call_and_return_conditional_losses_5496804

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
đ
Ž
F__inference_dense_154_layer_call_and_return_conditional_losses_5496851

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 


Ţ
/__inference_sequential_79_layer_call_fn_5497164

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity˘StatefulPartitionedCallŚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_sequential_79_layer_call_and_return_conditional_losses_54970002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:˙˙˙˙˙˙˙˙˙::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ó
ł
#__inference__traced_restore_5497452
file_prefix&
"assignvariableop_conv2d_148_kernel&
"assignvariableop_1_conv2d_148_bias(
$assignvariableop_2_conv2d_149_kernel&
"assignvariableop_3_conv2d_149_bias'
#assignvariableop_4_dense_154_kernel%
!assignvariableop_5_dense_154_bias'
#assignvariableop_6_dense_155_kernel%
!assignvariableop_7_dense_155_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_10
,assignvariableop_17_adam_conv2d_148_kernel_m.
*assignvariableop_18_adam_conv2d_148_bias_m0
,assignvariableop_19_adam_conv2d_149_kernel_m.
*assignvariableop_20_adam_conv2d_149_bias_m/
+assignvariableop_21_adam_dense_154_kernel_m-
)assignvariableop_22_adam_dense_154_bias_m/
+assignvariableop_23_adam_dense_155_kernel_m-
)assignvariableop_24_adam_dense_155_bias_m0
,assignvariableop_25_adam_conv2d_148_kernel_v.
*assignvariableop_26_adam_conv2d_148_bias_v0
,assignvariableop_27_adam_conv2d_149_kernel_v.
*assignvariableop_28_adam_conv2d_149_bias_v/
+assignvariableop_29_adam_dense_154_kernel_v-
)assignvariableop_30_adam_dense_154_bias_v/
+assignvariableop_31_adam_dense_155_kernel_v-
)assignvariableop_32_adam_dense_155_bias_v
identity_34˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_18˘AssignVariableOp_19˘AssignVariableOp_2˘AssignVariableOp_20˘AssignVariableOp_21˘AssignVariableOp_22˘AssignVariableOp_23˘AssignVariableOp_24˘AssignVariableOp_25˘AssignVariableOp_26˘AssignVariableOp_27˘AssignVariableOp_28˘AssignVariableOp_29˘AssignVariableOp_3˘AssignVariableOp_30˘AssignVariableOp_31˘AssignVariableOp_32˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9˘	RestoreV2˘RestoreV2_1Ž
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*ş
value°B­!B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesĐ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÓ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:::::::::::::::::::::::::::::::::*/
dtypes%
#2!	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_148_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_148_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp$assignvariableop_2_conv2d_149_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_149_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_154_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_154_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_155_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_155_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0	*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17Ľ
AssignVariableOp_17AssignVariableOp,assignvariableop_17_adam_conv2d_148_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18Ł
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_conv2d_148_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19Ľ
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_conv2d_149_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20Ł
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_conv2d_149_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21¤
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_154_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22˘
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_154_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23¤
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_155_kernel_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24˘
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_155_bias_mIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25Ľ
AssignVariableOp_25AssignVariableOp,assignvariableop_25_adam_conv2d_148_kernel_vIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26Ł
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_conv2d_148_bias_vIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27Ľ
AssignVariableOp_27AssignVariableOp,assignvariableop_27_adam_conv2d_149_kernel_vIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28Ł
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_conv2d_149_bias_vIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29¤
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_154_kernel_vIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30˘
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_154_bias_vIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31¤
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_155_kernel_vIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32˘
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_155_bias_vIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32¨
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp´
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33Á
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*
_input_shapes
: :::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: 

H
,__inference_flatten_79_layer_call_fn_5497175

inputs
identity¤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_flatten_79_layer_call_and_return_conditional_losses_54968322
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
÷1
ł
"__inference__wrapped_model_5496742
conv2d_148_input;
7sequential_79_conv2d_148_conv2d_readvariableop_resource<
8sequential_79_conv2d_148_biasadd_readvariableop_resource;
7sequential_79_conv2d_149_conv2d_readvariableop_resource<
8sequential_79_conv2d_149_biasadd_readvariableop_resource:
6sequential_79_dense_154_matmul_readvariableop_resource;
7sequential_79_dense_154_biasadd_readvariableop_resource:
6sequential_79_dense_155_matmul_readvariableop_resource;
7sequential_79_dense_155_biasadd_readvariableop_resource
identityá
.sequential_79/conv2d_148/Conv2D/ReadVariableOpReadVariableOp7sequential_79_conv2d_148_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype020
.sequential_79/conv2d_148/Conv2D/ReadVariableOpú
sequential_79/conv2d_148/Conv2DConv2Dconv2d_148_input6sequential_79/conv2d_148/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2!
sequential_79/conv2d_148/Conv2DŘ
/sequential_79/conv2d_148/BiasAdd/ReadVariableOpReadVariableOp8sequential_79_conv2d_148_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype021
/sequential_79/conv2d_148/BiasAdd/ReadVariableOpí
 sequential_79/conv2d_148/BiasAddBiasAdd(sequential_79/conv2d_148/Conv2D:output:07sequential_79/conv2d_148/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 sequential_79/conv2d_148/BiasAddŹ
sequential_79/conv2d_148/ReluRelu)sequential_79/conv2d_148/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_79/conv2d_148/Reluř
'sequential_79/max_pooling2d_148/MaxPoolMaxPool+sequential_79/conv2d_148/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2)
'sequential_79/max_pooling2d_148/MaxPoolâ
.sequential_79/conv2d_149/Conv2D/ReadVariableOpReadVariableOp7sequential_79_conv2d_149_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype020
.sequential_79/conv2d_149/Conv2D/ReadVariableOp
sequential_79/conv2d_149/Conv2DConv2D0sequential_79/max_pooling2d_148/MaxPool:output:06sequential_79/conv2d_149/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2!
sequential_79/conv2d_149/Conv2DŘ
/sequential_79/conv2d_149/BiasAdd/ReadVariableOpReadVariableOp8sequential_79_conv2d_149_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype021
/sequential_79/conv2d_149/BiasAdd/ReadVariableOpí
 sequential_79/conv2d_149/BiasAddBiasAdd(sequential_79/conv2d_149/Conv2D:output:07sequential_79/conv2d_149/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 sequential_79/conv2d_149/BiasAddŹ
sequential_79/conv2d_149/ReluRelu)sequential_79/conv2d_149/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_79/conv2d_149/Reluř
'sequential_79/max_pooling2d_149/MaxPoolMaxPool+sequential_79/conv2d_149/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2)
'sequential_79/max_pooling2d_149/MaxPool
sequential_79/flatten_79/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  2 
sequential_79/flatten_79/ConstÝ
 sequential_79/flatten_79/ReshapeReshape0sequential_79/max_pooling2d_149/MaxPool:output:0'sequential_79/flatten_79/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 sequential_79/flatten_79/Reshape×
-sequential_79/dense_154/MatMul/ReadVariableOpReadVariableOp6sequential_79_dense_154_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02/
-sequential_79/dense_154/MatMul/ReadVariableOpß
sequential_79/dense_154/MatMulMatMul)sequential_79/flatten_79/Reshape:output:05sequential_79/dense_154/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_79/dense_154/MatMulŐ
.sequential_79/dense_154/BiasAdd/ReadVariableOpReadVariableOp7sequential_79_dense_154_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.sequential_79/dense_154/BiasAdd/ReadVariableOpâ
sequential_79/dense_154/BiasAddBiasAdd(sequential_79/dense_154/MatMul:product:06sequential_79/dense_154/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
sequential_79/dense_154/BiasAddĄ
sequential_79/dense_154/ReluRelu(sequential_79/dense_154/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
sequential_79/dense_154/ReluÖ
-sequential_79/dense_155/MatMul/ReadVariableOpReadVariableOp6sequential_79_dense_155_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02/
-sequential_79/dense_155/MatMul/ReadVariableOpß
sequential_79/dense_155/MatMulMatMul*sequential_79/dense_154/Relu:activations:05sequential_79/dense_155/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
sequential_79/dense_155/MatMulÔ
.sequential_79/dense_155/BiasAdd/ReadVariableOpReadVariableOp7sequential_79_dense_155_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_79/dense_155/BiasAdd/ReadVariableOpá
sequential_79/dense_155/BiasAddBiasAdd(sequential_79/dense_155/MatMul:product:06sequential_79/dense_155/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
sequential_79/dense_155/BiasAddŠ
sequential_79/dense_155/SoftmaxSoftmax(sequential_79/dense_155/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
sequential_79/dense_155/Softmax}
IdentityIdentity)sequential_79/dense_155/Softmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:˙˙˙˙˙˙˙˙˙:::::::::a ]
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameconv2d_148_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ľ

č
/__inference_sequential_79_layer_call_fn_5496971
conv2d_148_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity˘StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallconv2d_148_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_sequential_79_layer_call_and_return_conditional_losses_54969522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:˙˙˙˙˙˙˙˙˙::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_nameconv2d_148_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ţ 
ˇ
J__inference_sequential_79_layer_call_and_return_conditional_losses_5496952

inputs
conv2d_148_5496928
conv2d_148_5496930
conv2d_149_5496934
conv2d_149_5496936
dense_154_5496941
dense_154_5496943
dense_155_5496946
dense_155_5496948
identity˘"conv2d_148/StatefulPartitionedCall˘"conv2d_149/StatefulPartitionedCall˘!dense_154/StatefulPartitionedCall˘!dense_155/StatefulPartitionedCall
"conv2d_148/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_148_5496928conv2d_148_5496930*
Tin
2*
Tout
2*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_148_layer_call_and_return_conditional_losses_54967542$
"conv2d_148/StatefulPartitionedCallü
!max_pooling2d_148/PartitionedCallPartitionedCall+conv2d_148/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_max_pooling2d_148_layer_call_and_return_conditional_losses_54967702#
!max_pooling2d_148/PartitionedCallŹ
"conv2d_149/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_148/PartitionedCall:output:0conv2d_149_5496934conv2d_149_5496936*
Tin
2*
Tout
2*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_149_layer_call_and_return_conditional_losses_54967882$
"conv2d_149/StatefulPartitionedCallü
!max_pooling2d_149/PartitionedCallPartitionedCall+conv2d_149/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_max_pooling2d_149_layer_call_and_return_conditional_losses_54968042#
!max_pooling2d_149/PartitionedCallŢ
flatten_79/PartitionedCallPartitionedCall*max_pooling2d_149/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_flatten_79_layer_call_and_return_conditional_losses_54968322
flatten_79/PartitionedCall
!dense_154/StatefulPartitionedCallStatefulPartitionedCall#flatten_79/PartitionedCall:output:0dense_154_5496941dense_154_5496943*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_154_layer_call_and_return_conditional_losses_54968512#
!dense_154/StatefulPartitionedCall
!dense_155/StatefulPartitionedCallStatefulPartitionedCall*dense_154/StatefulPartitionedCall:output:0dense_155_5496946dense_155_5496948*
Tin
2*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_155_layer_call_and_return_conditional_losses_54968782#
!dense_155/StatefulPartitionedCall
IdentityIdentity*dense_155/StatefulPartitionedCall:output:0#^conv2d_148/StatefulPartitionedCall#^conv2d_149/StatefulPartitionedCall"^dense_154/StatefulPartitionedCall"^dense_155/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:˙˙˙˙˙˙˙˙˙::::::::2H
"conv2d_148/StatefulPartitionedCall"conv2d_148/StatefulPartitionedCall2H
"conv2d_149/StatefulPartitionedCall"conv2d_149/StatefulPartitionedCall2F
!dense_154/StatefulPartitionedCall!dense_154/StatefulPartitionedCall2F
!dense_155/StatefulPartitionedCall!dense_155/StatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ţ 
ˇ
J__inference_sequential_79_layer_call_and_return_conditional_losses_5497000

inputs
conv2d_148_5496976
conv2d_148_5496978
conv2d_149_5496982
conv2d_149_5496984
dense_154_5496989
dense_154_5496991
dense_155_5496994
dense_155_5496996
identity˘"conv2d_148/StatefulPartitionedCall˘"conv2d_149/StatefulPartitionedCall˘!dense_154/StatefulPartitionedCall˘!dense_155/StatefulPartitionedCall
"conv2d_148/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_148_5496976conv2d_148_5496978*
Tin
2*
Tout
2*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_148_layer_call_and_return_conditional_losses_54967542$
"conv2d_148/StatefulPartitionedCallü
!max_pooling2d_148/PartitionedCallPartitionedCall+conv2d_148/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_max_pooling2d_148_layer_call_and_return_conditional_losses_54967702#
!max_pooling2d_148/PartitionedCallŹ
"conv2d_149/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_148/PartitionedCall:output:0conv2d_149_5496982conv2d_149_5496984*
Tin
2*
Tout
2*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_149_layer_call_and_return_conditional_losses_54967882$
"conv2d_149/StatefulPartitionedCallü
!max_pooling2d_149/PartitionedCallPartitionedCall+conv2d_149/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_max_pooling2d_149_layer_call_and_return_conditional_losses_54968042#
!max_pooling2d_149/PartitionedCallŢ
flatten_79/PartitionedCallPartitionedCall*max_pooling2d_149/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_flatten_79_layer_call_and_return_conditional_losses_54968322
flatten_79/PartitionedCall
!dense_154/StatefulPartitionedCallStatefulPartitionedCall#flatten_79/PartitionedCall:output:0dense_154_5496989dense_154_5496991*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_154_layer_call_and_return_conditional_losses_54968512#
!dense_154/StatefulPartitionedCall
!dense_155/StatefulPartitionedCallStatefulPartitionedCall*dense_154/StatefulPartitionedCall:output:0dense_155_5496994dense_155_5496996*
Tin
2*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_155_layer_call_and_return_conditional_losses_54968782#
!dense_155/StatefulPartitionedCall
IdentityIdentity*dense_155/StatefulPartitionedCall:output:0#^conv2d_148/StatefulPartitionedCall#^conv2d_149/StatefulPartitionedCall"^dense_154/StatefulPartitionedCall"^dense_155/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:˙˙˙˙˙˙˙˙˙::::::::2H
"conv2d_148/StatefulPartitionedCall"conv2d_148/StatefulPartitionedCall2H
"conv2d_149/StatefulPartitionedCall"conv2d_149/StatefulPartitionedCall2F
!dense_154/StatefulPartitionedCall!dense_154/StatefulPartitionedCall2F
!dense_155/StatefulPartitionedCall!dense_155/StatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
˙

+__inference_dense_154_layer_call_fn_5497195

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallŐ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dense_154_layer_call_and_return_conditional_losses_54968512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 

O
3__inference_max_pooling2d_149_layer_call_fn_5496810

inputs
identityÍ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*W
fRRP
N__inference_max_pooling2d_149_layer_call_and_return_conditional_losses_54968042
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ž(
á
J__inference_sequential_79_layer_call_and_return_conditional_losses_5497086

inputs-
)conv2d_148_conv2d_readvariableop_resource.
*conv2d_148_biasadd_readvariableop_resource-
)conv2d_149_conv2d_readvariableop_resource.
*conv2d_149_biasadd_readvariableop_resource,
(dense_154_matmul_readvariableop_resource-
)dense_154_biasadd_readvariableop_resource,
(dense_155_matmul_readvariableop_resource-
)dense_155_biasadd_readvariableop_resource
identityˇ
 conv2d_148/Conv2D/ReadVariableOpReadVariableOp)conv2d_148_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02"
 conv2d_148/Conv2D/ReadVariableOpĆ
conv2d_148/Conv2DConv2Dinputs(conv2d_148/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
conv2d_148/Conv2DŽ
!conv2d_148/BiasAdd/ReadVariableOpReadVariableOp*conv2d_148_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!conv2d_148/BiasAdd/ReadVariableOpľ
conv2d_148/BiasAddBiasAddconv2d_148/Conv2D:output:0)conv2d_148/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_148/BiasAdd
conv2d_148/ReluReluconv2d_148/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_148/ReluÎ
max_pooling2d_148/MaxPoolMaxPoolconv2d_148/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2
max_pooling2d_148/MaxPool¸
 conv2d_149/Conv2D/ReadVariableOpReadVariableOp)conv2d_149_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02"
 conv2d_149/Conv2D/ReadVariableOpâ
conv2d_149/Conv2DConv2D"max_pooling2d_148/MaxPool:output:0(conv2d_149/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingVALID*
strides
2
conv2d_149/Conv2DŽ
!conv2d_149/BiasAdd/ReadVariableOpReadVariableOp*conv2d_149_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!conv2d_149/BiasAdd/ReadVariableOpľ
conv2d_149/BiasAddBiasAddconv2d_149/Conv2D:output:0)conv2d_149/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_149/BiasAdd
conv2d_149/ReluReluconv2d_149/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
conv2d_149/ReluÎ
max_pooling2d_149/MaxPoolMaxPoolconv2d_149/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2
max_pooling2d_149/MaxPoolu
flatten_79/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  2
flatten_79/ConstĽ
flatten_79/ReshapeReshape"max_pooling2d_149/MaxPool:output:0flatten_79/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
flatten_79/Reshape­
dense_154/MatMul/ReadVariableOpReadVariableOp(dense_154_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_154/MatMul/ReadVariableOp§
dense_154/MatMulMatMulflatten_79/Reshape:output:0'dense_154/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_154/MatMulŤ
 dense_154/BiasAdd/ReadVariableOpReadVariableOp)dense_154_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_154/BiasAdd/ReadVariableOpŞ
dense_154/BiasAddBiasAdddense_154/MatMul:product:0(dense_154/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_154/BiasAddw
dense_154/ReluReludense_154/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_154/ReluŹ
dense_155/MatMul/ReadVariableOpReadVariableOp(dense_155_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02!
dense_155/MatMul/ReadVariableOp§
dense_155/MatMulMatMuldense_154/Relu:activations:0'dense_155/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_155/MatMulŞ
 dense_155/BiasAdd/ReadVariableOpReadVariableOp)dense_155_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_155/BiasAdd/ReadVariableOpŠ
dense_155/BiasAddBiasAdddense_155/MatMul:product:0(dense_155/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_155/BiasAdd
dense_155/SoftmaxSoftmaxdense_155/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_155/Softmaxo
IdentityIdentitydense_155/Softmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:˙˙˙˙˙˙˙˙˙:::::::::W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
é

,__inference_conv2d_149_layer_call_fn_5496798

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallđ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_conv2d_149_layer_call_and_return_conditional_losses_54967882
StatefulPartitionedCallŠ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ŢP
ű
 __inference__traced_save_5497341
file_prefix0
,savev2_conv2d_148_kernel_read_readvariableop.
*savev2_conv2d_148_bias_read_readvariableop0
,savev2_conv2d_149_kernel_read_readvariableop.
*savev2_conv2d_149_bias_read_readvariableop/
+savev2_dense_154_kernel_read_readvariableop-
)savev2_dense_154_bias_read_readvariableop/
+savev2_dense_155_kernel_read_readvariableop-
)savev2_dense_155_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_conv2d_148_kernel_m_read_readvariableop5
1savev2_adam_conv2d_148_bias_m_read_readvariableop7
3savev2_adam_conv2d_149_kernel_m_read_readvariableop5
1savev2_adam_conv2d_149_bias_m_read_readvariableop6
2savev2_adam_dense_154_kernel_m_read_readvariableop4
0savev2_adam_dense_154_bias_m_read_readvariableop6
2savev2_adam_dense_155_kernel_m_read_readvariableop4
0savev2_adam_dense_155_bias_m_read_readvariableop7
3savev2_adam_conv2d_148_kernel_v_read_readvariableop5
1savev2_adam_conv2d_148_bias_v_read_readvariableop7
3savev2_adam_conv2d_149_kernel_v_read_readvariableop5
1savev2_adam_conv2d_149_bias_v_read_readvariableop6
2savev2_adam_dense_154_kernel_v_read_readvariableop4
0savev2_adam_dense_154_bias_v_read_readvariableop6
2savev2_adam_dense_155_kernel_v_read_readvariableop4
0savev2_adam_dense_155_bias_v_read_readvariableop
savev2_1_const

identity_1˘MergeV2Checkpoints˘SaveV2˘SaveV2_1
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_61d3f9311857401da7cb6ce6dbe32b97/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename¨
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*ş
value°B­!B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesĘ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesŔ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_148_kernel_read_readvariableop*savev2_conv2d_148_bias_read_readvariableop,savev2_conv2d_149_kernel_read_readvariableop*savev2_conv2d_149_bias_read_readvariableop+savev2_dense_154_kernel_read_readvariableop)savev2_dense_154_bias_read_readvariableop+savev2_dense_155_kernel_read_readvariableop)savev2_dense_155_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_conv2d_148_kernel_m_read_readvariableop1savev2_adam_conv2d_148_bias_m_read_readvariableop3savev2_adam_conv2d_149_kernel_m_read_readvariableop1savev2_adam_conv2d_149_bias_m_read_readvariableop2savev2_adam_dense_154_kernel_m_read_readvariableop0savev2_adam_dense_154_bias_m_read_readvariableop2savev2_adam_dense_155_kernel_m_read_readvariableop0savev2_adam_dense_155_bias_m_read_readvariableop3savev2_adam_conv2d_148_kernel_v_read_readvariableop1savev2_adam_conv2d_148_bias_v_read_readvariableop3savev2_adam_conv2d_149_kernel_v_read_readvariableop1savev2_adam_conv2d_149_bias_v_read_readvariableop2savev2_adam_dense_154_kernel_v_read_readvariableop0savev2_adam_dense_154_bias_v_read_readvariableop2savev2_adam_dense_155_kernel_v_read_readvariableop0savev2_adam_dense_155_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 */
dtypes%
#2!	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardŹ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1˘
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesĎ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ă
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesŹ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ś
_input_shapes¤
Ą: :::::
::	:: : : : : : : : : :::::
::	::::::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::-)
'
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::% !

_output_shapes
:	: !

_output_shapes
::"

_output_shapes
: 
ň
Ž
F__inference_dense_155_layer_call_and_return_conditional_losses_5496878

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Á
c
G__inference_flatten_79_layer_call_and_return_conditional_losses_5496832

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs"ŻL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ć
serving_default˛
U
conv2d_148_inputA
"serving_default_conv2d_148_input:0˙˙˙˙˙˙˙˙˙=
	dense_1550
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:żî
Ŕ:
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
		variables

regularization_losses
trainable_variables
	keras_api

signatures
*z&call_and_return_all_conditional_losses
{_default_save_signature
|__call__"˛7
_tf_keras_sequential7{"class_name": "Sequential", "name": "sequential_79", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_79", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_148", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_148", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_149", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_149", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_79", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_154", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_155", "trainable": true, "dtype": "float32", "units": 26, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_79", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_148", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_148", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_149", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_149", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_79", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_154", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_155", "trainable": true, "dtype": "float32", "units": 26, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Ç


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*}&call_and_return_all_conditional_losses
~__call__"˘	
_tf_keras_layer	{"class_name": "Conv2D", "name": "conv2d_148", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "stateful": false, "config": {"name": "conv2d_148", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
á
	variables
regularization_losses
trainable_variables
	keras_api
*&call_and_return_all_conditional_losses
__call__"Ń
_tf_keras_layerˇ{"class_name": "MaxPooling2D", "name": "max_pooling2d_148", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d_148", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ě	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ľ
_tf_keras_layer{"class_name": "Conv2D", "name": "conv2d_149", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_149", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 128]}}
â
	variables
regularization_losses
 trainable_variables
!	keras_api
+&call_and_return_all_conditional_losses
__call__"Ń
_tf_keras_layerˇ{"class_name": "MaxPooling2D", "name": "max_pooling2d_149", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d_149", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ç
"	variables
#regularization_losses
$trainable_variables
%	keras_api
+&call_and_return_all_conditional_losses
__call__"ś
_tf_keras_layer{"class_name": "Flatten", "name": "flatten_79", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten_79", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Ř

&kernel
'bias
(	variables
)regularization_losses
*trainable_variables
+	keras_api
+&call_and_return_all_conditional_losses
__call__"ą
_tf_keras_layer{"class_name": "Dense", "name": "dense_154", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_154", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3200]}}
Ř

,kernel
-bias
.	variables
/regularization_losses
0trainable_variables
1	keras_api
+&call_and_return_all_conditional_losses
__call__"ą
_tf_keras_layer{"class_name": "Dense", "name": "dense_155", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_155", "trainable": true, "dtype": "float32", "units": 26, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
ă
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmkmlmm&mn'mo,mp-mqvrvsvtvu&vv'vw,vx-vy"
	optimizer
X
0
1
2
3
&4
'5
,6
-7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
&4
'5
,6
-7"
trackable_list_wrapper
Ę
7layer_regularization_losses
		variables
8non_trainable_variables

regularization_losses
9layer_metrics
:metrics
trainable_variables

;layers
|__call__
{_default_save_signature
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
,:*2conv2d_148/kernel
:2conv2d_148/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
<layer_regularization_losses
	variables
=non_trainable_variables
regularization_losses
>layer_metrics
?metrics
trainable_variables

@layers
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ž
Alayer_regularization_losses
	variables
Bnon_trainable_variables
regularization_losses
Clayer_metrics
Dmetrics
trainable_variables

Elayers
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
-:+2conv2d_149/kernel
:2conv2d_149/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
Flayer_regularization_losses
	variables
Gnon_trainable_variables
regularization_losses
Hlayer_metrics
Imetrics
trainable_variables

Jlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Klayer_regularization_losses
	variables
Lnon_trainable_variables
regularization_losses
Mlayer_metrics
Nmetrics
 trainable_variables

Olayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Player_regularization_losses
"	variables
Qnon_trainable_variables
#regularization_losses
Rlayer_metrics
Smetrics
$trainable_variables

Tlayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
$:"
2dense_154/kernel
:2dense_154/bias
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
°
Ulayer_regularization_losses
(	variables
Vnon_trainable_variables
)regularization_losses
Wlayer_metrics
Xmetrics
*trainable_variables

Ylayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
#:!	2dense_155/kernel
:2dense_155/bias
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
°
Zlayer_regularization_losses
.	variables
[non_trainable_variables
/regularization_losses
\layer_metrics
]metrics
0trainable_variables

^layers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
_0
`1"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ť
	atotal
	bcount
c	variables
d	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}

	etotal
	fcount
g
_fn_kwargs
h	variables
i	keras_api"ż
_tf_keras_metric¤{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
.
a0
b1"
trackable_list_wrapper
-
c	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
e0
f1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
1:/2Adam/conv2d_148/kernel/m
#:!2Adam/conv2d_148/bias/m
2:02Adam/conv2d_149/kernel/m
#:!2Adam/conv2d_149/bias/m
):'
2Adam/dense_154/kernel/m
": 2Adam/dense_154/bias/m
(:&	2Adam/dense_155/kernel/m
!:2Adam/dense_155/bias/m
1:/2Adam/conv2d_148/kernel/v
#:!2Adam/conv2d_148/bias/v
2:02Adam/conv2d_149/kernel/v
#:!2Adam/conv2d_149/bias/v
):'
2Adam/dense_154/kernel/v
": 2Adam/dense_154/bias/v
(:&	2Adam/dense_155/kernel/v
!:2Adam/dense_155/bias/v
ö2ó
J__inference_sequential_79_layer_call_and_return_conditional_losses_5497122
J__inference_sequential_79_layer_call_and_return_conditional_losses_5497086
J__inference_sequential_79_layer_call_and_return_conditional_losses_5496895
J__inference_sequential_79_layer_call_and_return_conditional_losses_5496922Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ń2î
"__inference__wrapped_model_5496742Ç
˛
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *7˘4
2/
conv2d_148_input˙˙˙˙˙˙˙˙˙
2
/__inference_sequential_79_layer_call_fn_5497019
/__inference_sequential_79_layer_call_fn_5496971
/__inference_sequential_79_layer_call_fn_5497143
/__inference_sequential_79_layer_call_fn_5497164Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ś2Ł
G__inference_conv2d_148_layer_call_and_return_conditional_losses_5496754×
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *7˘4
2/+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
2
,__inference_conv2d_148_layer_call_fn_5496764×
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *7˘4
2/+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ś2ł
N__inference_max_pooling2d_148_layer_call_and_return_conditional_losses_5496770ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
2
3__inference_max_pooling2d_148_layer_call_fn_5496776ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
§2¤
G__inference_conv2d_149_layer_call_and_return_conditional_losses_5496788Ř
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *8˘5
30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
2
,__inference_conv2d_149_layer_call_fn_5496798Ř
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *8˘5
30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ś2ł
N__inference_max_pooling2d_149_layer_call_and_return_conditional_losses_5496804ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
2
3__inference_max_pooling2d_149_layer_call_fn_5496810ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ń2î
G__inference_flatten_79_layer_call_and_return_conditional_losses_5497170˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ö2Ó
,__inference_flatten_79_layer_call_fn_5497175˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
đ2í
F__inference_dense_154_layer_call_and_return_conditional_losses_5497186˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ő2Ň
+__inference_dense_154_layer_call_fn_5497195˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
đ2í
F__inference_dense_155_layer_call_and_return_conditional_losses_5497206˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ő2Ň
+__inference_dense_155_layer_call_fn_5497215˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
=B;
%__inference_signature_wrapper_5497050conv2d_148_inputŤ
"__inference__wrapped_model_5496742&',-A˘>
7˘4
2/
conv2d_148_input˙˙˙˙˙˙˙˙˙
Ş "5Ş2
0
	dense_155# 
	dense_155˙˙˙˙˙˙˙˙˙Ý
G__inference_conv2d_148_layer_call_and_return_conditional_losses_5496754I˘F
?˘<
:7
inputs+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ľ
,__inference_conv2d_148_layer_call_fn_5496764I˘F
?˘<
:7
inputs+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ţ
G__inference_conv2d_149_layer_call_and_return_conditional_losses_5496788J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "@˘=
63
0,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ś
,__inference_conv2d_149_layer_call_fn_5496798J˘G
@˘=
;8
inputs,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "30,˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙¨
F__inference_dense_154_layer_call_and_return_conditional_losses_5497186^&'0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
+__inference_dense_154_layer_call_fn_5497195Q&'0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙§
F__inference_dense_155_layer_call_and_return_conditional_losses_5497206],-0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 
+__inference_dense_155_layer_call_fn_5497215P,-0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙­
G__inference_flatten_79_layer_call_and_return_conditional_losses_5497170b8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 
,__inference_flatten_79_layer_call_fn_5497175U8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙ń
N__inference_max_pooling2d_148_layer_call_and_return_conditional_losses_5496770R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "H˘E
>;
04˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 É
3__inference_max_pooling2d_148_layer_call_fn_5496776R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ";84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ń
N__inference_max_pooling2d_149_layer_call_and_return_conditional_losses_5496804R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "H˘E
>;
04˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 É
3__inference_max_pooling2d_149_layer_call_fn_5496810R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ";84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ę
J__inference_sequential_79_layer_call_and_return_conditional_losses_5496895|&',-I˘F
?˘<
2/
conv2d_148_input˙˙˙˙˙˙˙˙˙
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 Ę
J__inference_sequential_79_layer_call_and_return_conditional_losses_5496922|&',-I˘F
?˘<
2/
conv2d_148_input˙˙˙˙˙˙˙˙˙
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 Ŕ
J__inference_sequential_79_layer_call_and_return_conditional_losses_5497086r&',-?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 Ŕ
J__inference_sequential_79_layer_call_and_return_conditional_losses_5497122r&',-?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ˘
/__inference_sequential_79_layer_call_fn_5496971o&',-I˘F
?˘<
2/
conv2d_148_input˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙˘
/__inference_sequential_79_layer_call_fn_5497019o&',-I˘F
?˘<
2/
conv2d_148_input˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙
/__inference_sequential_79_layer_call_fn_5497143e&',-?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙
/__inference_sequential_79_layer_call_fn_5497164e&',-?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙Â
%__inference_signature_wrapper_5497050&',-U˘R
˘ 
KŞH
F
conv2d_148_input2/
conv2d_148_input˙˙˙˙˙˙˙˙˙"5Ş2
0
	dense_155# 
	dense_155˙˙˙˙˙˙˙˙˙