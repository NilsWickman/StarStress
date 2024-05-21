import matplotlib.pyplot as plt
import numpy as np


x = np.array([11.829751968383789,10.285714149475098,9.95035171508789,9.96466064453125,9.846364974975586,
              9.485840797424316,9.35477352142334,9.27185344696045,9.465190887451172,9.340238571166992,9.136214256286621,
              9.065849304199219,8.973468780517578,9.068672180175781,8.925248146057129,8.946724891662598,8.825166702270508,
              8.829911231994629,8.922164916992188,8.787126541137695,8.616537094116211,8.549060821533203,8.644834518432617,
              8.462242126464844,8.50837230682373,8.376822471618652,8.802655220031738,8.574968338012695,8.340555191040039,
              8.404714584350586,8.542028427124023,8.44625473022461,8.238862037658691,8.397473335266113,8.666814804077148,
              8.419961929321289,8.387944221496582,8.449545860290527,8.372483253479004,8.276384353637695,8.445176124572754,
              8.373046875,8.398737907409668,8.47193431854248,7.947677135467529,8.307268142700195,8.198347091674805,8.1217041015625,
              8.077886581420898,8.367225646972656,8.344741821289062,7.9923601150512695,7.904350757598877,8.022221565246582,
              8.375152587890625,8.259859085083008,8.361854553222656,7.951716899871826,8.235921859741211,8.103482246398926,
              8.199838638305664,8.266338348388672,8.12094497680664,8.331493377685547,8.37442398071289,8.228020668029785,
              7.960366725921631,8.299962043762207,8.242576599121094,8.09521198272705,8.074686050415039,8.015715599060059,
              7.919968128204346,7.743879318237305,7.996562480926514,8.167142868041992,8.165905952453613,7.932444095611572,
              7.977149486541748,8.004229545593262,8.265483856201172,8.224709510803223,8.068924903869629,8.065053939819336,
              8.008811950683594,8.14254379272461,8.045265197753906,8.108991622924805,8.021841049194336,8.07831859588623,
              8.17316722869873,8.074612617492676,7.930159568786621,7.893863201141357,7.990840911865234,7.845008373260498,
              7.913803577423096,7.819501876831055,8.226672172546387,8.002886772155762,7.816262722015381,7.885004997253418,
              8.033418655395508,7.957968235015869,7.758327484130859,7.915035724639893,8.196560859680176,7.965023517608643,
              7.935573101043701,8.013687133789062,7.945009708404541,7.868428707122803,8.028149604797363,7.974345684051514,
              7.996659755706787,8.07182502746582,7.565547943115234,7.927074432373047,7.833132266998291,7.762467861175537,
              7.722447395324707,8.011151313781738,8.067489624023438,7.843471050262451,7.901343822479248,
              8.21023178100586,8.071364402770996,8.167162895202637,7.736806869506836,8.014933586120605,7.894509792327881,
              7.984460353851318,8.051759719848633,7.886662006378174,8.099282264709473,8.139394760131836,8.013615608215332,
              7.742434978485107,8.068338394165039,7.763594627380371,7.771348476409912,7.9566874504089355,7.955472469329834,
              7.728008270263672,7.77728271484375,7.8121137619018555,8.063129425048828,8.019959449768066,7.867209434509277,
              7.86838960647583,7.807732582092285,7.942162990570068,7.856908798217773,7.915060997009277,7.829991817474365,
              7.8918304443359375,7.980287075042725,7.888277530670166,7.741131782531738,7.715386867523193,7.811641216278076,
              7.668127059936523,7.734297752380371,7.645199298858643,8.045526504516602,7.828007698059082,7.6353325843811035,
              7.7078962326049805,7.8643293380737305,7.782169342041016,7.593319416046143,7.745796203613281,8.027018547058105,
              7.795362949371338,7.769193649291992,7.784751892089844, 7.612154960632324, 7.607800483703613, 7.794641494750977, 
              7.790273666381836, 7.570964813232422, 7.623302459716797, 7.658927917480469, 7.906196594238281, 7.8641510009765625, 
              7.715934753417969, 7.717048645019531, 7.65032958984375, 7.7845001220703125, 7.701316833496094, 7.7660369873046875, 
              7.687828063964844, 7.7410888671875, 7.82928466796875, 7.7353515625, 7.5899505615234375, 7.57379150390625, 7.6652374267578125, 
              7.525482177734375, 7.5960845947265625, 7.5103759765625, 7.902496337890625, 7.6903076171875, 7.498748779296875, 7.5767669677734375, 
              7.726806640625, 7.64862060546875, 7.4527130126953125, 7.60931396484375, 7.88568115234375, 7.667022705078125, 7.628875732421875, 
              7.7159423828125, 7.7445068359375, 7.682952880859375, 7.83343505859375, 7.784751892089844, 7.612154960632324, 7.607800483703613, 
              7.794641494750977, 7.790273666381836, 7.570964813232422, 7.623302459716797, 7.658927917480469, 7.906196594238281, 7.8641510009765625, 
              7.715934753417969, 7.717048645019531, 7.65032958984375, 7.7845001220703125, 7.701316833496094, 7.7660369873046875, 7.687828063964844, 
              7.7410888671875, 7.82928466796875, 7.7353515625, 7.5899505615234375, 7.57379150390625, 7.6652374267578125, 7.525482177734375, 7.5960845947265625, 
              7.5103759765625, 7.902496337890625, 7.6903076171875, 7.498748779296875, 7.5767669677734375, 7.726806640625, 7.64862060546875, 7.4527130126953125, 
              7.60931396484375, 7.88568115234375, 7.667022705078125, 7.628875732421875, 7.7159423828125, 7.7445068359375, 7.682952880859375, 7.83343505859375, 
              7.692014694213867,7.511385917663574,7.496644496917725,7.6798248291015625,7.674933910369873,7.461093902587891,7.519217491149902,7.544898509979248,
              7.798377990722656,7.757621765136719,7.609809398651123,7.605076313018799,7.538482666015625,7.678692817687988,7.598265647888184,7.663647651672363,
              7.588589191436768,7.636051177978516,7.719827651977539,7.626919746398926,7.483074188232422,7.470627784729004,7.5626020431518555,7.419827461242676,
              7.499863147735596,7.414872646331787,7.800410270690918,7.619956016540527,7.423736572265625,7.3942084312438965,7.563784122467041,7.562121391296387,
              7.387246131896973,7.151628017425537,7.104644298553467,7.298220157623291,7.298087120056152,7.402272701263428,7.450262546539307,7.466365814208984,
              7.715372562408447,7.667789936065674,7.518056392669678,7.514944076538086,7.4442057609558105,7.579716205596924,7.500985622406006,7.567790508270264,
              7.49907112121582,7.534078598022461,7.614676475524902,7.523284435272217,7.388363361358643,7.372890472412109,7.46513557434082,7.322333812713623,
              7.403461933135986,7.318004131317139,7.706013202667236,7.487863063812256,7.443532943725586,7.52352237701416,7.665622711181641,7.58148193359375,
              7.389206886291504,7.534061908721924,7.81809139251709,7.60153341293335,7.565151691436768,7.648121356964111,7.632236480712891,7.578293800354004,
              7.7295074462890625,7.671575546264648,7.762510299682617,7.834700107574463,7.348306179046631,7.706417560577393,7.6161580085754395,7.542522430419922,
              7.502261638641357,7.775894641876221,7.797568321228027,7.447170257568359,7.357452869415283,7.491333961486816,7.8193359375,7.7160258293151855,
              7.830193519592285,7.402202129364014,7.6936492919921875,7.591451168060303,7.691377639770508,7.76378059387207,7.601284980773926,7.816059112548828,
              7.865626811981201,7.748499393463135,7.482745170593262,7.803677558898926,7.757986068725586,7.650318622589111,7.503208160400391,7.311713218688965,
              7.270705699920654,7.4369354248046875,7.438436985015869,7.279492378234863,7.337099552154541,7.362222194671631,7.615008354187012,7.569053649902344,
              7.424409866333008,7.425814151763916,7.351341724395752,7.486563205718994,7.416754245758057,7.476979732513428,7.41231632232666,7.453724384307861,7.534141540527344,7.4356584548950195,7.296945571899414,7.296017169952393,7.376716136932373,7.2420148849487305,7.323323726654053,7.240945816040039,7.618291854858398,7.405887603759766,7.272703170776367,7.357229232788086,7.501507759094238,7.418070316314697,7.224942207336426,7.366472244262695,7.660023212432861,7.442784309387207,7.399542331695557,7.491674423217773,7.444272041320801,7.406202793121338,7.549152851104736,7.491344928741455,7.549185752868652,7.6199822425842285,7.1462788581848145,7.507723331451416,7.423445224761963,7.349832057952881,7.3076252937316895,7.576513767242432,7.605068683624268,7.255012035369873,7.162797451019287,7.285858631134033,7.626232624053955,7.513657569885254,7.6348347663879395,7.196866512298584,7.501026153564453,7.3993635177612305,7.496988296508789,7.568443775177002,7.400813579559326,7.6248321533203125,7.663404941558838,7.565232753753662,7.296840190887451,7.259346008300781,7.214584827423096,7.375808238983154,7.374557018280029,7.211263179779053,7.276618003845215,7.302262783050537,7.553785800933838,7.509757041931152,7.362246513366699,7.280773639678955,7.038454532623291,6.959880352020264,7.117624282836914,7.109364032745361,6.971658706665039,7.044549465179443,7.068255424499512,7.32016658782959,7.290637016296387,7.134117603302002,7.425314903259277,7.336161136627197,7.468291759490967,7.391268253326416,7.447173118591309,7.386399269104004,7.422884941101074,7.505561351776123,7.405820369720459])

x2 = np.array([11.81149959564209,10.216902732849121,9.905754089355469,9.938990592956543,9.817438125610352,9.46212387084961,9.371087074279785,9.27346134185791,9.448686599731445,9.335034370422363,9.138227462768555,9.063807487487793,8.98316478729248,9.087455749511719,8.946897506713867,8.976798057556152,8.851033210754395,8.853165626525879,8.951400756835938,8.821991920471191,8.641749382019043,8.568300247192383,8.66210651397705,8.485408782958984,8.539936065673828,8.402789115905762,8.831113815307617,8.600870132446289,8.366427421569824,8.426665306091309,8.563457489013672,8.479058265686035,8.269914627075195,8.42981243133545,8.694466590881348,8.448784828186035,8.413223266601562,8.470808982849121,8.390079498291016,8.291861534118652,8.463829040527344,8.380428314208984,8.413790702819824,8.488569259643555,7.965933322906494,8.313891410827637,8.202805519104004,8.139272689819336,8.085131645202637,8.365601539611816,8.35741901397705,8.00064754486084,7.9150495529174805,8.039043426513672,8.38437557220459,8.257451057434082,8.36356258392334,7.946551322937012,8.235118865966797,8.107548713684082,8.198782920837402,8.26467514038086,8.110430717468262,8.332296371459961,8.374723434448242,8.225866317749023,7.963688373565674,8.299068450927734,8.22217082977295,8.082523345947266,8.059281349182129,7.9968953132629395,8.111115455627441,8.093619346618652,7.973515510559082,7.954495429992676,8.0703763961792,8.200647354125977,8.06334400177002,7.86503791809082,7.765790939331055,8.221336364746094,8.030220985412598,8.115147590637207,7.742424011230469,7.879359722137451,7.946736812591553,8.096878051757812,8.12174129486084,8.132833480834961,7.97359037399292,7.9848151206970215,8.13139533996582,7.999180316925049,7.8297224044799805,7.806237697601318,8.07226848602295,7.772974967956543,7.829033374786377,7.938340187072754,7.84248685836792,7.880128383636475,8.231261253356934,7.912482261657715,7.991689205169678,8.172242164611816,7.8650102615356445,7.727606773376465,8.134493827819824,8.085795402526855,7.825926303863525,8.024703979492188,7.960928440093994,7.769406318664551,7.801629066467285,7.786708831787109,7.9653449058532715,7.570364952087402,7.5555644035339355,7.734777927398682,7.9912495613098145,7.7580108642578125,7.79490852355957,7.830855369567871,8.078070640563965,8.033184051513672,7.88519811630249,7.892549514770508,7.8269362449646,7.964803218841553,7.868961334228516,7.933689594268799,7.846996784210205,7.90728759765625,7.998721599578857,7.910356521606445,7.756899833679199,7.740177154541016,7.83253812789917,7.688407897949219,7.747972011566162,7.66429328918457,8.0613431930542,7.848551273345947,7.661710739135742,7.736328125,7.875771522521973,7.800319671630859,7.6042914390563965,7.761230945587158,8.04345989227295,7.813814640045166,7.7854390144348145,7.866226673126221,7.797338485717773,7.725525379180908,7.884487628936768,7.828943252563477,7.856753349304199,7.927740573883057,7.4385247230529785,7.791245460510254,7.7057600021362305,7.631460189819336,7.589560031890869,7.869654655456543,7.8826704025268555,7.526255130767822,7.439172267913818,7.573324203491211,7.906126022338867,7.795962810516357,7.909815788269043,7.486835479736328,7.780899524688721,7.680163860321045,7.7702460289001465,7.848443508148193,7.684686660766602,7.9043288230896,7.945173740386963,7.830466270446777,7.572687149047852,7.887086868286133,7.825536251068115,7.702406883239746,7.684332370758057,7.629233360290527,7.7361741065979,7.734588146209717,7.605937480926514,7.591007232666016,7.719030380249023,7.839657306671143,7.715763092041016,7.533358097076416,7.430952548980713,7.880641937255859,7.6945109367370605,7.773487091064453,7.414830684661865,7.5540452003479,7.634778022766113,7.779604911804199,7.808662414550781,7.817273139953613,7.661141872406006,7.652390480041504,7.819505214691162,7.693684101104736,7.535572052001953,7.513494491577148,7.777620792388916,7.473611831665039,7.542835235595703,7.6516337394714355,7.55189847946167,7.592754364013672,7.941927433013916,7.6235809326171875,7.701322078704834,7.888189792633057,7.597376346588135,7.456329822540283,7.8671135902404785,7.815853595733643,7.555319309234619,7.759954452514648,7.6991753578186035,7.524893283843994,7.670645713806152,7.5903520584106445,7.763667106628418,7.6367387771606445,7.8229522705078125,7.58671760559082,7.625578880310059,7.5830793380737305,7.720466136932373,7.56914758682251,7.724551200866699,7.66060733795166,7.778609275817871,7.671200275421143,7.762049198150635,7.6100053787231445,7.631702423095703,7.6950225830078125,7.762413501739502,7.539821624755859,7.389150619506836,7.607774257659912,7.590492248535156,7.572443008422852,7.701515197753906,7.429279327392578,7.667977333068848,7.652966499328613,7.63778018951416,7.559307098388672,7.724850177764893,7.694226264953613,7.769009113311768,7.662753105163574,7.883021831512451,7.889261245727539,7.766230583190918,7.664413928985596,7.553411960601807,7.753993511199951,7.5180745124816895,7.645441055297852,7.7277374267578125,7.539859771728516,7.636963844299316,7.852634429931641,7.636330604553223,7.634639263153076,7.70456600189209,7.676063537597656,7.864195346832275,7.757626533508301,7.713018894195557,7.699701309204102,7.6214280128479,7.476907730102539,7.819509506225586,7.770101070404053,7.6379714012146,7.852581024169922,7.543104648590088,7.667967319488525,7.575096130371094,7.683355331420898,7.682674407958984,7.593755722045898,7.55756139755249,7.630664348602295,7.61644172668457,7.561600208282471,7.4775824546813965,7.461675643920898,7.636938095092773,7.650193691253662,7.4378132820129395,7.4923906326293945,7.526447772979736,7.78131103515625,7.733115196228027,7.590114593505859,7.599715709686279,7.541182994842529,7.658607482910156,7.579208850860596,7.646732807159424,7.575017929077148,7.629972457885742,7.7078776359558105,7.630969047546387,7.4792256355285645,7.470110893249512,7.5598320960998535,7.425605773925781,7.487399578094482,7.4144134521484375,7.793936252593994,7.592051982879639,7.408777713775635,7.478791236877441,7.621711730957031,7.5572967529296875,7.353728771209717,7.500278949737549,7.796036243438721,7.568239688873291,7.53366756439209,7.623414039611816,7.553896903991699,7.4907917976379395,7.638341903686523,7.590370178222656,7.616074562072754,7.6969990730285645,7.202154636383057,7.5667524337768555,7.480276584625244,7.408565998077393,7.37055778503418,7.637977123260498,7.659191131591797,7.308590888977051,7.219340801239014,7.356980800628662,7.6864752769470215,7.37971830368042,7.363995552062988,7.532596588134766,7.5431036949157715,7.330193042755127,7.385620594024658,7.422134876251221,7.674860000610352,7.623852729797363,7.481236457824707,7.489571571350098,7.421754360198975,7.544224739074707,7.470705032348633,7.5369415283203125,7.465091228485107,7.522716045379639,7.595791339874268,7.516163349151611,7.367044925689697,7.357893466949463,7.447441101074219,7.314370632171631,7.382368564605713,7.306455135345459,7.687621593475342,7.4817609786987305,7.299581050872803,7.376636981964111,7.517627716064453,7.451838970184326,7.247027397155762,7.385049819946289,7.688164710998535,7.458191871643066,7.425447463989258,7.515135765075684,7.445251941680908,7.388724327087402,7.535393714904785,7.48610782623291,7.510953426361084,7.586210250854492,7.0957207679748535,7.460441589355469,7.378002643585205,7.462273597717285,7.300504207611084,7.273775100708008,7.441462993621826,7.454853057861328,7.242971420288086,7.301084518432617,7.327487468719482,7.5854597091674805,7.53264856338501,7.392541408538818,7.401968002319336,7.327366828918457,7.453455924987793,7.384692668914795,7.448519706726074,7.385051727294922,7.434917449951172,7.506137847900391,7.421435832977295,7.277073860168457,7.272860527038574,7.35872220993042,7.223014831542969,7.296589374542236,7.221564292907715,7.599239349365234,7.389634609222412,7.209526538848877,7.293609619140625,7.433815956115723,7.364740371704102,7.160270690917969,7.294166564941406,7.598548412322998,7.373279571533203,7.338022708892822,7.433671474456787,7.355850696563721,7.308139801025391,7.450896739959717,7.403472900390625,7.427207946777344,7.501145362854004,7.01221227645874,7.376140594482422,7.300046443939209,7.334321022033691,7.293768882751465,7.556864261627197,7.580019474029541,7.232999801635742,7.142158508300781,7.269720077514648,7.611359119415283,7.558770179748535

])
#print(len(x))
y_list = []
y = 0
for iteration in x:
    y += (128*32)
    y_list.append(y)
print(y)
y_list = np.array(y_list)

print(len(x))
print(len(y_list))
print(len(x2))

plt.plot(y_list, x)
plt.plot(y_list, x2)
plt.xlabel("Observation Action Pairs")
plt.ylabel("Loss")
plt.ylim(0, 12)

plt.savefig('Baseline_Loss.jpeg')
