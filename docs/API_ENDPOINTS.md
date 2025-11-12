# NeuroLab API Endpoints Documentation

Complete reference for all available API endpoints in the NeuroLab EEG Analysis platform.

## Base URL
```
http://localhost:8000
```

## Authentication

Most endpoints require JWT authentication. Include the token in the Authorization header:
```
Authorization: Bearer <your_jwt_token>
```

---

## Core Endpoints (main.py)

### 1. Health Check
**GET** `/health`

Check the health status of the API and model availability.

**Response:**
```json
{
  "status": "healthy",
  "diagnostics": {
    "model_loaded": true,
    "tensorflow_available": true
  }
}
```

**Status Codes:**
- `200`: Service is healthy
- `503`: Service unavailable

---

### 2. Root / API Info
**GET** `/`

Get basic API information and available endpoints.

**Response:**
```json
{
  "name": "NeuroLab EEG Analysis API",
  "version": "1.0.0",
  "description": "API for EEG signal processing and mental state classification",
  "endpoints": {
    "health": "/health",
    "upload": "/upload",
    "analyze": "/analyze",
    "calibrate": "/calibrate",
    "recommendations": "/recommendations"
  }
}
```

---

### 3. Upload EEG File
**POST** `/upload`

Upload and process EEG data files (CSV, EDF, BDF formats).

**Tags:** Analysis

**Request:**
- **Form Data:**
  - `file`: EEG data file (multipart/form-data)
  - `encrypt_response`: boolean (optional, default: false)

**OR**

- **JSON Body:**
```json
{
  "alpha": 0.5,
  "beta": 0.3,
  "theta": 0.2,
  "delta": 0.1,
  "gamma": 0.4
}
```

**Response:**
```json
{
  "predicted_state": [0, 1, 2],
  "temporal_analysis": {...},
  "cognitive_metrics": {...},
  "clinical_recommendations": {...}
}
```

**Status Codes:**
- `200`: Success
- `400`: Invalid file or data
- `500`: Processing error

---

### 4. Analyze EEG Data
**POST** `/analyze`

Analyze EEG data and return mental state classification.

**Tags:** Analysis

**Request Body:**
```json
{
  "alpha": 0.5,
  "beta": 0.3,
  "theta": 0.2,
  "delta": 0.1,
  "gamma": 0.4,
  "subject_id": "SUBJ001",
  "session_id": "SESS001"
}
```

**Response:**
```json
{
  "predicted_state": 1,
  "confidence": 0.92,
  "state_label": "stressed",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Status Codes:**
- `200`: Success
- `400`: Invalid data
- `422`: Validation error
- `500`: Processing error

---

### 5. Calibrate Model
**POST** `/calibrate`

Calibrate the model with new subject-specific data.

**Tags:** Model

**Request Body:**
```json
{
  "X_train": [[0.1, 0.2, 0.3, 0.4, 0.5]],
  "y_train": [0],
  "subject_id": "SUBJ001"
}
```

**Response:**
```json
{
  "status": "calibration_started",
  "message": "Calibration process initiated"
}
```

**Status Codes:**
- `200`: Calibration started
- `500`: Calibration error
- `503`: Model no.cc
rolabneups://docs. htttation:enocum
- Dlab.ccrt@neuromail: suppo
- Edellab_morg/neurour-o/yob.comithub: https://gtHus:
- Gir issuestions o

For que
## Support

---
;
```log(result)e.();
consolse.jsonsponeReanalyzawait t = ulonst res
});
c001'
  })n_id: 'SESS  sessio1',
  UBJ00_id: 'Sectubj0.4,
    s
    gamma: 0.1,a:  delt0.2,
     theta: 
  a: 0.3,
    betalpha: 0.5,({
    ingifySON.str J
  body:},  n}`
tokeess_er ${accarion': `Beuthorizat',
    'An/jsonpplicatiope': 'aontent-Ty    'C
: { headersT',
 OS  method: 'P', {
yze000/analst:8hoaltp://locetch('htit fawaonse = splyzeReonst anaEG data
cnalyze E

// Anse.json();Respologin await token } =t { access_ns
co  })
});sword123'
: 'pasordssw
    pame: 'user',erna
    usringify({: JSON.st
  body,on' }plication/jspe': 'apTytent- { 'Conrs: headeT',
 d: 'POS {
  methogin',auth/lo/api/t:8000://localhosetch('http f= awaitginResponse const lo
ginLo/ cript
/vasjacript
```## JavaS
```

#
  }'ESS001": "Session_id"    "s",
UBJ001": "S_idect"subj 0.4,
    amma": "g1,
   ": 0.  "delta0.2,
  "theta":   0.3,
   a":
    "bet": 0.5,pha"al    -d '{
 N" \
 _TOKEOURr Yion: BeareAuthorizat
  -H "n" \ion/jso applicatType:ent-nt
  -H "Colyze \0/anaost:800/localhPOST http:/ -X ata
curllyze EEG dAna

# }'3"d12sworpasd":","passworer"use":"ernam
  -d '{"usn" \sopplication/jType: aontent-
  -H "Cth/login \0/api/auhost:800local://ttpOST hrl -X Pogin
cu``bash
# LURL
`
### c```
e.json())
int(responsders
)
praders=heaheta,
    n=da,
    jso00/analyze"alhost:80://lochttpst(
    " requests.po
response ="
}S001 "SES":_idssion
    "se"SUBJ001", d":subject_i0.4,
    ""gamma": : 0.1,
    "delta",
    0.2ta": "the   : 0.3,
 ta""be 0.5,
    ":lpha= {
    "aata "}
d {token}"Bearer ftion":rizautho"A {aders = data
heEEGze 
# Analyoken"]
ss_ton()["accee.jsespons = ren3"}
)
tokd12sswor": "pa, "passwordr"me": "usen={"userna
    jsogin",louth/8000/api/aalhost:loc   "http://.post(
 questsponse = ren
res

# Logirt requestsmpoython
ihon
```pPyt

### mplesode Exa
## C

---
c0/redot:800os://localh http**ReDoc**:
- t:8000/docsosttp://localhr UI**: hSwaggeon:
- **ntatidocumeI tive APteraccess in
Acn
entatioive Documactnter
---

## Ittentive
used/Aoc`2`: Fs
- iouressed/Anx`1`: St- laxed/Calm
`: Rels
- `0e Labetal Stat)

### Men0-100 Hz power (3 wavemma`: Gamma
- `ga-4 Hz) power (0.5 wavelta`: Delta)
- `deower (4-8 Hza wave petta`: Th `the
- (13-30 Hz)werve poa`: Beta wa- `bet Hz)
wer (8-13ave poa w`: Alph`alphare Names
-  FeatuEEG

### ct Notation ObjeaScript*JSON**: JavFormat
- *mi Data BDF**: BioSet
- **n Data Forma**: EuropeaEDFues
- **rated valmma-sepaCo- **CSV**: s
 File Formatupportedats

### Srmta Fo

## Da--`

-``642248000
mit-Reset: 19
X-RateLiRemaining: teLimit-it: 10
X-Rait-Limim``
X-RateLponses:
`esuded in rers are inclit head

Rate limgitinlim rate int**: Nolth endpo **Heast of 5
-ond, burts/secrequesint**: 2  endpo*Upload 20
- * ofond, burstests/secrequ10 : ndpoints**General e**abuse:

- revent imited to pate-lpoints are r

API endtingimi

## Rate L-ble

--ce UnavailaServi`: 03
- `5rerver Erroal S`: Internor
- `500idation err - Valble Entityrocessa- `422`: Unpists
ready ex Resource allict -`: Confst
- `409n't exirce doesou - Resnd4`: Not Fou`40sions
-  permisntie Insufficdden -03`: Forbi
- `4onicatintvalid authe or iningzed - Misshori Unaut
- `401`:d input data- Invalit  Reques400`: Bad
- `reated1`: Ccess
- `20: Suc
- `200`Codes
s atu## Common St```

#wrong"
}
went t ing wharib descsage "Error mes"detail":json
{
  
``` format:
winglloin the foes sponsrror re return eaydpoints m

All enResponsesor # Err-

#

--ror`: Er00- `5uthorized
01`: Unaess
- `4 `200`: Succ:**
-Codestus **Sta
```

t_001"
}liennt clieed for cer clear"Buffessage": ss",
  "ms": "succe{
  "statuson
`jse:**
``espon
```

**R
}t_001"encli": "client_idson
{
  "y:**
```jest Bodqu**Re

n (required)arer tokeion`: Be`Authorizats:**
- eader*Hnt.

*ific clier a specng buffer fomistreathe 

Clear tream/clear`i/sT** `/apPOSuffer
**ar Stream B. Cle-

### 12

--g error: Processin`500`
- rizednautho`401`: U
- lid data: Inva
- `400`cess- `200`: Sucs Codes:**
tatud

**Sues allowealNaN or Inf vNo 200 μV
- plitude: am
- Maximum 0el: 1000s per channsampleimum Max: 64
- nelsmum chan
- Maxies:**n Rullidatio

**Va```
}

  }  }": 0.23
  el_3chann
      "2": 0.32,channel_"     45,
 : 0.annel_1" "ch     nce": {
ure_importa "feat
   ","limeod":     "methility": {
interpretab"
  : false,ted" "encrypZ",
 10:30:00024-01-15Tstamp": "22,
  "timee_ms": 45.ing_timprocess "": 0.85,
 nfidence: 1,
  "conant_state""domi1, 2],
  0, es": [tat"predicted_s
  son
{nse:**
```j`

**Respoalse
}
``ity": fterpretabil"include_inse,
  nse": fal_resporypt  "ence,
": trurtifactsclean_a"
  lstm",hanced_cnn_": "enel_type "mod",
 ent_001": "cli_id  "client.7]
  ],
.5, 0.6, 0.3, 0.4, 0.6],
    [05, 04, 0..3, 0. 02,
    [0..4, 0.5],.3, 0[0.1, 0.2, 0a": [
    "eeg_dat
  ```json
{dy:**
uest Bo*Required)

*req token (`: Bearerionhorizat `Autrs:**
-
**Headeta.
EG daeaming Eime str real-t
Processtream`
 `/api/s**POST**
G Datatream EE11. Sy)

### oint.pndpg_eamin/strepoints (apiing End
## Stream---
s

dy existrname alrea- `409`: Useadmin)
n (not  Forbidde`:- `403thorized
Unau`:  `401d
-er createUs201`: 
- ` Codes:**atus

**St"
}
```T10:30:00Z"2024-01-15": _atreated"c
  r"],": ["usesle"ro
  ",ewuser": "nrnameuse{
  "json
*
```onse:*

**Resp
}
```ser"]oles": ["u23",
  "repassword1"secur sword":"paser",
   "newuse":"usernamn
{
  
```jso**dy:t BoRequese

**dmin rolken with aarer ton`: Beuthorizatioers:**
- `A*Headole).

*min r adresunt (requi accoerew usCreate a n
ers`
api/auth/usPOST** `/nly)
**er (Admin Oeate Us. Cr
### 10-

--horized
`: Unaut401
- `t successfulouLog
- `200`:  Codes:****Status
}
```

ully"ccessfsuout ": "Logged sage  "mes",
successus": "statn
{
  ":**
```jso**Responseal)

ionate (optalid to invtokenh fresoken`: Reresh_t
- `refed)quirer token (reation`: Bear`Authoriz
- ers:****Headens.

tokrefresh invalidate 
Logout and ogout`
/api/auth/lPOST** `r Logout
** Use
### 9.

---
 tokeneshd refrd or expire: Invali
- `401`edfresh0`: Token re*
- `20des:*
**Status Co``

  }
}
`"]sers": ["ule    "ro",
serrname": "u
    "useinfo": {
  "user_3600,_in":  "expiresarer",
 e": "beoken_typ",
  "tken_refresh_token": "newtosh_
  "refretoken",ew_access_oken": "ness_taccson
{
  "`j*
``onse:*Resp```

**
}
"enh_tokrefresn": "your_tokesh_"refre
  n
{```jso Body:**

**Request token.
shefreen using ress tokesh accfresh`

Reuth/refri/aPOST** `/apoken
**esh T8. Refr## ---

#dentials

Invalid cre: 
- `401`successful Login `200`:**
- s:Status Code}
```

**]
  }
"user"[oles":    "ruser",
 : "ername"   "us: {
 info""user_  0,
360pires_in": ,
  "exr"earetype": "b"token_",
  ecure_token: "random_ssh_token"efre",
  "rXVCJ9...kpsInR5cCI6IiJIUzI1NiIGciO"eyJhben": s_tok  "acces``json
{
se:**
`on*Resp}
```

*123"
 "passwordd":"passworuser",
  ame": "ern
{
  "usson*
```jst Body:*
**Reques.
tokenceive JWT  reser andnticate uAuthegin`

louth/* `/api/aOST*in
**P7. User Log## 
#uth.py)
s (api/an Endpointenticatio

## Auth---ndations

meving recometrieError r500`: ccess
- `00`: Su*
- `2es:*Codus *Stat```

*  ]
}
s"
exercisething eaep brce deti"Prac,
    evels"uce stress ledk to r a brear takingside"Con
    tions": [endacomm"re",
  "SUBJ001": subject_id"S001",
  : "SESn_id"sessio{
  "
``jsonsponse:**
``

**Re01
``UBJ0t_id=S001&subjecsion_id=SESSns?sesndatioGET /recomme
le:**
```

**Examp identifier): Subject (requiredt_id`bjec- `sudentifier
ion i: Sessequired)(r` idession_
- `ss:**ry Parameter**Quenalysis


**Tags:** A
 analysis.reviousn pons based oecommendatiized rrsonalGet pens`

mendatio* `/recomns
**GET*endatioecomm6. Get R
---

### e
ailablt av