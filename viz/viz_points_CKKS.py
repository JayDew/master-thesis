import matplotlib.pyplot as plt
import numpy as np

OPT_X = [1]
OPT_Y = [0]
plt.scatter(OPT_X, OPT_Y, color='orange', marker='*', s=500, label='optimal solution')  # plot optimal solution as a star
# Generate a range of x values
x = np.linspace(0, 1, 400)
# Compute the corresponding y values
y = -x + 1
plt.plot(x, y, label='Ax = b')

alpha = 0.1

CKKS = [(0.5005925823964491, 0.49941648502311853), (0.5011699129660683, 0.49918836851453546), (0.5014144725609854, 0.4988724138804831), (0.5020267928760928, 0.49840155161921884), (0.5026536112216856, 0.4977810004028675), (0.5034273337242128, 0.49707064459140654), (0.5032909450860111, 0.49623016135427667), (0.5041054930916259, 0.4961593596545942), (0.5053017119756448, 0.4951831994789077), (0.505901262404457, 0.4940990254736612), (0.5065353618141589, 0.4936963074439555), (0.5070524550757289, 0.49272555389603284), (0.5082432357094754, 0.49174155747662796), (0.5093388110163681, 0.49068019663947127), (0.5104232606531727, 0.4897956272976489), (0.5112344065274073, 0.48897362829072816), (0.5115578468134516, 0.4883071146430943), (0.51209556470506, 0.4880838909268024), (0.512355842861582, 0.4877067320090191), (0.5128843075633144, 0.48688389036091634), (0.5146086290319756, 0.4861663448383355), (0.5153568204200238, 0.4844431822175067), (0.5163903732084354, 0.48361678208745656), (0.5165291713186526, 0.4827256742554693), (0.5174323261154602, 0.48180067704959767), (0.5191091372097343, 0.48111616281124525), (0.51996548346538, 0.4801290780373964), (0.5209265576086379, 0.4793662732429694), (0.5224335012693335, 0.4780247404663349), (0.5238583038686502, 0.47645089463596424), (0.5246596652044359, 0.47484178974193947), (0.5259442375826844, 0.4737721351794694), (0.5270549752913096, 0.47308340117490905), (0.5272714312798302, 0.47272123037120783), (0.5272414218402843, 0.47233129024575016), (0.5283041079671339, 0.471877475719946), (0.5292878069114696, 0.47088928480992187), (0.5303329387943742, 0.469538488516852), (0.5314389470117118, 0.46866918032409594), (0.5323269063098703, 0.46802720225077993), (0.5333619797810566, 0.4668388229047656), (0.5342510035683583, 0.4658264073444954), (0.5350731532150009, 0.4647706139579115), (0.5364102091685301, 0.4643218825261427), (0.5369574564643322, 0.46299503020435856), (0.538449025143597, 0.4618254375151568), (0.5397402642147436, 0.46012434572230243), (0.5406044697315001, 0.4588200326425891), (0.5415866387246125, 0.4585940296320985), (0.5418524119194211, 0.4573329927277976), (0.5437528094171867, 0.45654810824164715), (0.5446073348176048, 0.45478259719264086), (0.5456627565460572, 0.45447467940134084), (0.5461862172900936, 0.4533368090301311), (0.5471734075774038, 0.45216744088342453), (0.5483783230437602, 0.4515107122424969), (0.5493792700120509, 0.4504031104116434), (0.5506950402865182, 0.4497773117579011), (0.5514475468402454, 0.4488717921474741), (0.5526192415374439, 0.4474881605805422), (0.5532320587775458, 0.44614196205228124), (0.554312939862556, 0.4458333527746773), (0.5556547916419043, 0.4445760869232257), (0.5566308911279477, 0.44325467250581013), (0.5570448670732011, 0.44266848438619155), (0.5578177594602789, 0.4427873874531222), (0.5583805190603225, 0.44143526334489613), (0.559694947565665, 0.4405093038968573), (0.5607303588762405, 0.4393660187095761), (0.5617513036462751, 0.438069613282358), (0.5630435139887776, 0.4370473939627425), (0.5646009138404481, 0.43595223672816263), (0.5656186415862756, 0.43473317603914985), (0.5661616720184994, 0.4333310327656274), (0.5675965929149014, 0.4324785393961788), (0.5686761122834617, 0.4310615193117627), (0.5700533537231277, 0.43037011812251924), (0.5708363907344425, 0.42966694013144713), (0.5714857390864038, 0.4287518818930453), (0.5724212944374291, 0.4273037445322737), (0.5743114252627407, 0.4264805683872381), (0.5751475298817803, 0.424399761475616), (0.5764274101294053, 0.42341887399592243), (0.5771995954024726, 0.4219166940276614), (0.5785333141201959, 0.42093367991027486), (0.5797579768536802, 0.42012057114106244), (0.5811559860485501, 0.41854670457873006), (0.5830047910422148, 0.41708592066934097), (0.5842537453506141, 0.41635217380469935), (0.5851555917190123, 0.4155680316306279), (0.5854964582344392, 0.4138137776173885), (0.5866733527661652, 0.41303277445300945), (0.5876018785575363, 0.4122546423607741), (0.5887742183784913, 0.4120166455488317), (0.5894856893065602, 0.41067799529839444), (0.5906184234798342, 0.40978459386633087), (0.592065781188446, 0.40877855184765505), (0.5933209921412548, 0.4070154220573268), (0.5943983051756859, 0.40535761765910744), (0.5953257116740146, 0.404153266425976), (0.5965481927014289, 0.40345843480765076), (0.5980606645718227, 0.4026656849534514), (0.5988230366581896, 0.40114570106201175), (0.600183361453459, 0.40009178827392533), (0.6014348995282853, 0.3987348383777715), (0.6028250542772491, 0.39699177080208253), (0.6045970938728671, 0.39585213795990426), (0.6057227497367782, 0.39355717595577167), (0.6074115509176643, 0.3917897183422373), (0.6091828200715603, 0.3917740988507384), (0.6098823293635117, 0.39048796580816547), (0.6113222834396621, 0.38875460304854337), (0.6131036786982592, 0.38737597555463393), (0.614291475444261, 0.3851636558250117), (0.6160826808222503, 0.3838441577990124), (0.6175528366601272, 0.382741096693625), (0.618324709567522, 0.3816235298417557), (0.6183909987190795, 0.3808693644731124), (0.6195062004089198, 0.3803532237073453), (0.6209079723206156, 0.3796703618570756), (0.6216827954456418, 0.3784183395583056), (0.622812221656553, 0.3773339479795944), (0.624574527070787, 0.37618498142489337), (0.6253702925563536, 0.37431006087166196), (0.6261924191490168, 0.3732900446566115), (0.6274244465236313, 0.37227436415718873), (0.6290790643965661, 0.3707533499795035), (0.6308421585408868, 0.3694366798462018), (0.6313407958865846, 0.3678925265381642), (0.6323385795225962, 0.3673702494306196), (0.6331353976597657, 0.36715172963994075), (0.6336771390134275, 0.366281550835366), (0.6349657083261253, 0.36544858131456165), (0.6358509443197299, 0.36360434245897527), (0.6373734061226215, 0.36278084471483446), (0.6389509238197837, 0.3617632934776798), (0.640374655915732, 0.3594933262391818), (0.6421666829166556, 0.35866930217785004), (0.6426176780184754, 0.3569058716199648), (0.6446902130981249, 0.35653316500377086), (0.6446129250526726, 0.3550144243762434), (0.6456424951268842, 0.3548288405312031), (0.6463856598531968, 0.3538976912529822), (0.6475186241305377, 0.3518309202871075), (0.6489784929289951, 0.3501934480425011), (0.6500687775115965, 0.349451231176057), (0.6509068218957724, 0.34844853325327496), (0.6519926728597512, 0.3484925578771478), (0.651475281957499, 0.3472757496619603), (0.6528582381815479, 0.3473379929000889), (0.6531209803133805, 0.3470343423839089), (0.6538863000559688, 0.34656252286270234), (0.654691392533343, 0.3451438911707852), (0.6559301014229604, 0.3440020543987109), (0.6566938782267044, 0.3431091671526896), (0.6576306758846288, 0.342980205888967), (0.6588279197225043, 0.34191003394036285), (0.6592195602678309, 0.3401501475908255), (0.6604246947603124, 0.3395607008896898), (0.6612663851401859, 0.33854491449883145), (0.6623484884528219, 0.3371319595706766), (0.6637284928250271, 0.3363297244016494), (0.6642212965388565, 0.3356066263552095), (0.665148569937673, 0.3346150330980146), (0.666255053029502, 0.33333752715777215), (0.6675400783827808, 0.331934470744201), (0.6690484670412217, 0.33060842300884413), (0.6709156357011258, 0.3296474889547775), (0.6724085341368271, 0.32791788885209583), (0.673106259428516, 0.32634602769805365), (0.6745785649839414, 0.32539567290440236), (0.6760593374789721, 0.3233755238240326), (0.6776806097627913, 0.32237349919479635), (0.6787675040856763, 0.3217631965716178), (0.6794545990416727, 0.32084634372884047), (0.6804725047459629, 0.3198915196612478), (0.6814921486468056, 0.31835087592628286), (0.6826385659790807, 0.3175851652917632), (0.6831094241326268, 0.3166626526632159), (0.6851524209724493, 0.3162142529384671), (0.6855343024755008, 0.31414207577545017), (0.687213622166261, 0.31322278935921954), (0.6883894129627415, 0.3115530894825521), (0.6903000843433542, 0.31055910509909584), (0.6917515031747898, 0.30788195090384735), (0.6934948514887912, 0.3056838556288138), (0.6952800600793774, 0.30491621300579813), (0.6962720296453271, 0.30334559131336025), (0.6982444685068165, 0.3025048142026733), (0.6995721567387725, 0.30105617658099826), (0.7003900288416178, 0.29906579655361654), (0.7019274251602143, 0.29802567346565845), (0.7028687774532344, 0.29698820783226704), (0.7044736502891981, 0.29621597806424543), (0.7057881551295488, 0.2945424824771109), (0.7065172083819188, 0.2932677969555008), (0.7069612564451465, 0.2915462276964023), (0.7091245334420585, 0.2908696420293494), (0.7103114960362433, 0.28981391053391625), (0.711040999883193, 0.288583977475472), (0.7130675537993352, 0.2878337540595183), (0.7134505032835261, 0.2860389868947453), (0.7147073080375828, 0.28504623225740877), (0.716195869865633, 0.2837440526840188), (0.7170086655203263, 0.2828434160142636), (0.7179106001070613, 0.2818544292953707), (0.7185995256181023, 0.2811258456102332), (0.719314389419321, 0.2812520614174454), (0.7197133965504826, 0.28032561966857195), (0.7205291320139102, 0.2788426682176569), (0.72190734305687, 0.27819524381282534), (0.7222346438549867, 0.2774469373521574), (0.723502461103852, 0.2771974793726884), (0.7243394419509049, 0.2755603683162157), (0.7253708262080804, 0.27446318813947845), (0.7256697872282577, 0.2739954329856289), (0.7261692313638092, 0.2731714314383554), (0.7272701122551095, 0.2725164066975637), (0.7283049899522117, 0.2716894637118775), (0.729785015539526, 0.2705399570543483), (0.7307624901932296, 0.2687529004904738), (0.7326570497384907, 0.2681942240720518), (0.7329312878741966, 0.2664199580044887), (0.7344939687680035, 0.265775631283803), (0.7352936505929265, 0.26478028064693737), (0.7361526675560232, 0.2637792851448752), (0.7370499921465961, 0.26342768231423236), (0.7374946558640784, 0.2626425528137688), (0.7381547059974466, 0.2615620662381121), (0.7392261356989785, 0.2608991757550089), (0.7405145506575481, 0.2593259460012649), (0.7420065137455326, 0.25796108775497945), (0.7431638281476141, 0.2571514079832622), (0.7446550973047243, 0.2557429177113355), (0.7463571817628385, 0.2537796959364561), (0.7472076071018151, 0.2524231897594612), (0.7478460182432698, 0.2521029661766625), (0.7488737162392294, 0.2513946123876671), (0.7501145005373355, 0.25013122198678966), (0.7520095659556725, 0.24917939342468362), (0.7526860503577698, 0.24740082429491586), (0.7543068750451797, 0.2462581987032552), (0.755619157050342, 0.24464357018265756), (0.7561357900063913, 0.24345517480953652), (0.7576125408550577, 0.2432024295141536), (0.7586281012451026, 0.24139789941876608), (0.7600592828351851, 0.24039075982597682), (0.7607205173799044, 0.23957985121496692), (0.7613533160023375, 0.23824780933400586), (0.762919856806096, 0.23711887849243152), (0.7647363049815997, 0.2357139020566451), (0.7661552207098371, 0.2341092300243402), (0.7667672900746566, 0.2329150286307956), (0.7678065632021548, 0.23217298915127602), (0.7687747467040542, 0.2312710243634612), (0.7697771452629985, 0.23031452955378343), (0.7702917503695039, 0.2286962496062375), (0.7717223420204261, 0.22841594833851384), (0.7727558918594996, 0.2272292468526901), (0.7739044801087475, 0.22654163696497598), (0.7746084508966606, 0.22542237515506056), (0.7761621910994216, 0.2242861441888712), (0.7769069668745109, 0.22317300962523903), (0.7774324429842121, 0.22283043117579893), (0.7778602604392507, 0.22253640859322427), (0.778496638985725, 0.2219103984485474), (0.7792278338617827, 0.22017733881971108), (0.7806806687586704, 0.21925487630097282), (0.7818439141268565, 0.21784550898511493), (0.7835593946505044, 0.21665507594079236), (0.7848302750023147, 0.21561162011457663), (0.7856046444413147, 0.21475852299413203), (0.7863574225622774, 0.2140615453071608), (0.7879341279862027, 0.21272343586154066), (0.7887243227428412, 0.21115677235669816), (0.7897943626185747, 0.21036840841171078), (0.7902682107499872, 0.20912061160274525), (0.7905082735955989, 0.20879037883968055), (0.7915319166015502, 0.20875882876567567), (0.7924884686029652, 0.2084524127374194), (0.7928241960852588, 0.20716072479562475), (0.7935716020518556, 0.2062486263897218), (0.7944628913242399, 0.20515673487716102), (0.7952913230310319, 0.20352603416108078), (0.7967842086991163, 0.20290566351296724), (0.797985541488492, 0.20194236898414036), (0.7986461089622101, 0.20068962880582722), (0.7997402937526408, 0.20030495012206925), (0.8006408091107775, 0.1992116399565293), (0.8012120763745636, 0.19801062424817523), (0.8024306921169745, 0.19718568858069635), (0.8040120594207912, 0.19561183578206573), (0.8053622144205441, 0.1949253768122993), (0.8065800759672164, 0.1940446930544018), (0.808313031873393, 0.19280731247915078), (0.8084019755913424, 0.1907348735634216), (0.8100502755334796, 0.19010012947374433), (0.8111039816236687, 0.18908151281101193), (0.8118389451683329, 0.18840098027591012), (0.8130797310546204, 0.18758051435514023), (0.8140587170221413, 0.18631871262113703), (0.8150542648065201, 0.1850838143104517), (0.8164376496575998, 0.18442827261611883), (0.8173043525950784, 0.18293811353223619), (0.8189038591962867, 0.18142281378929181), (0.8203755678602052, 0.18017080579496), (0.8212623950864807, 0.17887325339692958), (0.8222918076117912, 0.1777554485746732), (0.8235755799995372, 0.17661179352788253), (0.8244097030120272, 0.17525492644921167), (0.8254377979456488, 0.17459618682989933), (0.8266393553679656, 0.17374917817695024), (0.8272490820780168, 0.17221257106344584), (0.8282951502774253, 0.17153212543314836), (0.8289141352550778, 0.1706961423982267), (0.8299623423017884, 0.17007851077672362), (0.8315677393679032, 0.1687926982273307), (0.8329756899671233, 0.16710005386200977), (0.8342028178597477, 0.16553995190549714), (0.8360882652138786, 0.16439195977695117), (0.8374225159678528, 0.163114672098146), (0.8380712495307339, 0.16146992486261538), (0.8391718630077154, 0.16036603406024824), (0.8406589469456925, 0.15979019006788536), (0.8413693287327195, 0.15892313621014403), (0.8424645038537543, 0.15763953515942686), (0.8436306904278161, 0.15604143725858333), (0.8442857289166282, 0.1554548197567298), (0.8455627966018188, 0.15509212215067708), (0.8461741848186601, 0.15302419225808986), (0.8479313457975928, 0.15193401708347137), (0.8488180144105628, 0.15046157038137212), (0.8500416751363249, 0.1498157314821039), (0.8512220083147451, 0.14878019616135557), (0.8525802197481233, 0.14759921583549232), (0.8542413520783902, 0.146691172886713), (0.8552098110150629, 0.14496393003090932), (0.8564939226002987, 0.14349029700947466), (0.8573317248407043, 0.14216525463941987), (0.8582840335666949, 0.1417516799277434), (0.8591690557499556, 0.14123010832740265), (0.8598736584645607, 0.13993686377287184), (0.8612380368247472, 0.13931478327920643), (0.8615137778691875, 0.1378533800591575), (0.8632991274588938, 0.1378735299303198), (0.8639214474241378, 0.1364047065936533), (0.8655813007548321, 0.1348537354507315), (0.8668630431782712, 0.13296911671184833), (0.8686436332292209, 0.13194403028819843), (0.8701065798526583, 0.13050161997189383), (0.8710767090212145, 0.12924842542401688), (0.871843896146816, 0.12833189243386783), (0.872577162417739, 0.1275884275602739), (0.8734225240873961, 0.12675901854564317), (0.8746439537845699, 0.12561430396062886), (0.8753161027301984, 0.12404566886592279), (0.8772050039404681, 0.12341011544023056), (0.8782867204865603, 0.12144155160120836), (0.879463654014358, 0.11986335222076012), (0.8810200683665271, 0.1185255314948364), (0.8830562841943155, 0.11712403665428237), (0.8841468205652474, 0.1153198221963487), (0.8858231320460331, 0.11398693390597316), (0.8871061570248452, 0.11244829225605664), (0.8880941342476053, 0.11203617856634714), (0.8889080316004543, 0.11065650720233731), (0.8904634696834877, 0.10946876967466655), (0.8921949902518433, 0.10787395074676087), (0.8938698110915904, 0.10663901782115386), (0.8948513828672808, 0.10521463972560036), (0.8965737367348039, 0.10405574906751078), (0.8976289235262546, 0.10212619939570562), (0.8989598898546268, 0.10098110633013549), (0.9003731413190929, 0.09960949386842248), (0.9022612291495604, 0.09782115509333802), (0.9028942612581291, 0.09585850146148148), (0.9044424716763396, 0.09582794490819126), (0.9051629549408465, 0.09469092855106814), (0.9066558999008628, 0.09372092762587264), (0.9075768232263935, 0.09223001568396164), (0.9088338701634876, 0.09106319708818933), (0.9101018282129378, 0.08969508511343023), (0.9115421253056031, 0.088295775956988), (0.9127745734195569, 0.08690832080511586), (0.9144246917313305, 0.08561173173368784), (0.9154377479270905, 0.08371565719426967), (0.9174455613985221, 0.08315291394020223), (0.9181774738544057, 0.08171169533193841), (0.9192109390226896, 0.08101083202683296), (0.9199600939255961, 0.07958882804702672), (0.9211728601208905, 0.0784667902624017), (0.9223817594650732, 0.07769880263486843), (0.9228886524938262, 0.07643201557927141), (0.9245919669860391, 0.07547681946671322), (0.9258744203631182, 0.07355065769529663), (0.9278812907971734, 0.07275594329223846), (0.9285604295515777, 0.07152288531810878), (0.929491741549526, 0.07015836152733158), (0.930438220362849, 0.06910198045851142), (0.9316397891754307, 0.06866117222063181), (0.9323567663824489, 0.06703229356866677), (0.93395373198412, 0.06623051758118594), (0.9355648221342988, 0.06506025412416663), (0.9366572949813887, 0.0631913578774996), (0.9382933849284147, 0.06246735214835827), (0.9390906781822494, 0.06017550315975037), (0.9414902266279404, 0.05907604904812971), (0.942396416036144, 0.0577867769510344), (0.9429917547730867, 0.05667306271824716), (0.9442027772253286, 0.05553421540283372), (0.9455904598042543, 0.05446556659454946), (0.9457919596337072, 0.05280155768297982), (0.9475235649958687, 0.05249632575572553), (0.9480849745625393, 0.05149365151429985), (0.9491782155597859, 0.05092010799836827), (0.9502894570231472, 0.049436778623337584), (0.9512480112872315, 0.04811812395672413), (0.9527907957341774, 0.04714819466913709), (0.9543375830539707, 0.04627395770328792), (0.9547095423370414, 0.045278676359443326), (0.955351660068447, 0.04480972525406457), (0.955625470662727, 0.04415823527259632), (0.9560566634653482, 0.04343226662996685), (0.956823174358679, 0.042142337859146034), (0.9584755454140363, 0.04157466568263634), (0.9601302602915158, 0.04035704239126286), (0.9608624363718689, 0.038695552830969086), (0.961963223166014, 0.03804362513912186), (0.9635611334561036, 0.036695011398385366), (0.9650883490437963, 0.03534954743374371), (0.96584801119757, 0.03380883725545213), (0.9669296642033947, 0.03234490390817989), (0.968856402653006, 0.030847677290905953), (0.970207382835169, 0.028671560775382746), (0.9720611980431022, 0.027948706064345182), (0.9738015996659276, 0.026789815960871655), (0.9746908550795814, 0.02537527770535063), (0.9750045953391213, 0.02432173937194667), (0.9763255661887209, 0.02344712220213238), (0.9777735775542606, 0.022683553499910546), (0.9786914370568797, 0.021041012954322796), (0.9806396081752539, 0.019779363434689383), (0.9819541319379141, 0.018937875430330674), (0.9822040479963368, 0.017241400798783647), (0.9836852083684935, 0.01615695263609017), (0.9856182848740802, 0.014733922674856936), (0.9880201647053585, 0.012820403316185483), (0.9892767498565613, 0.011192715353323295), (0.9901506803320812, 0.009395001439223099), (0.9925165105273369, 0.0077884561077835235), (0.993986209732085, 0.005367249608568953), (0.9956912225900754, 0.0042485575759684206), (0.9967236906868716, 0.003452156170591726), (0.9980743498537339, 0.0025991092772505416), (0.9992655139965039, 0.0003639197507680607), (1.0016071147222958, 0.0), (1.0026027757117195, 0.0), (1.0021971410834596, 0.0), (1.0014138567627002, 0.0), (1.0008439143388086, 0.0), (1.0008513926609521, 0.0), (1.001127262068697, 0.0), (1.0008408425709698, 0.0), (1.000755321682229, 0.0), (1.0006823049478208, 0.0), (1.0008926210425049, 0.0), (1.0012662105939498, 0.0), (1.0013797232594404, 0.0), (1.0014690365670529, 0.0), (1.0014101212432132, 0.0), (1.0012889391762014, 0.0), (1.0011351471651373, 0.0), (1.0010530785632448, 0.0), (1.000966128644363, 0.0), (1.001262710532226, 0.0), (1.0018123081743162, 0.0), (1.0015303761821948, 0.0), (1.001501955185815, 0.0), (1.001596333807724, 0.0), (1.0012269681044148, 0.0), (1.0007867894729798, 0.0), (1.0009438813390876, 0.0), (1.0008915202076931, 0.0), (1.0006800202902166, 0.0), (1.0005942745920418, 0.0), (1.0015756816273627, 0.0), (1.0024772163181646, 0.0), (1.0018642831414315, 0.0), (1.0014373442368771, 0.0), (1.0007811888627542, 0.0), (1.0009872564409608, 0.0), (1.0009078831595248, 0.0), (1.0006021261549032, 0.0), (1.0011007316093055, 0.0), (1.0011477573035388, 0.0), (1.0013027154273957, 0.0), (1.000751646152939, 0.0), (1.001123133880163, 0.0), (1.002067060997971, 0.0), (1.0018734377574514, 0.0), (1.0015215196171356, 0.0), (1.001413359247136, 0.0), (1.001454002159681, 0.0), (1.0010492943265352, 0.0), (1.001070085987948, 0.0), (1.0012956603477179, 0.0), (1.0019115749747391, 0.0), (1.0018292074627524, 0.0), (1.0016806322630951, 0.0), (1.001666783765647, 0.0), (1.0013042780378367, 0.0), (1.0013267667503878, 0.0), (1.0008068960924021, 0.0), (1.0012399411841593, 0.0), (1.0014304007624077, 0.0), (1.0012875875135612, 0.0), (1.0014599362114933, 0.0), (1.0012883527482868, 0.0), (1.0016340266568158, 0.0), (1.0016905746034717, 0.0), (1.0014230838831113, 0.0), (1.0012409486385376, 0.0), (1.0012015679559387, 0.0), (1.0016418249289563, 0.0), (1.0016644446346041, 0.0), (1.0012783039257427, 0.0), (1.0013992028210232, 0.0), (1.0009600866217494, 0.0), (1.0011821338442217, 0.0), (1.0012803225571127, 0.0), (1.001429468041158, 0.0), (1.0018587472666656, 0.0), (1.0011282413526457, 0.0), (1.0011776634155019, 0.0), (1.001348433009042, 0.0), (1.0010081484483968, 0.0), (1.0013474821685528, 0.0), (1.001932893968888, 0.0), (1.0008933054891367, 0.0), (0.999999958631142, 0.0), (1.0006092634512205, 0.0), (1.0010828703285901, 0.0), (1.0010209742443765, 0.0), (1.0016062221089599, 0.0), (1.0014138022313415, 0.0), (1.0005341662395, 0.0), (1.0009804068719048, 0.0), (1.0014900312931847, 0.0), (1.0010007296544399, 0.0), (1.0005168741763268, 0.0), (1.0010978177906316, 0.0), (1.0014975611239587, 0.0), (1.0010106835098735, 0.0), (1.0009936406628828, 0.0), (1.0015081849732956, 0.0), (1.0020357371701216, 0.0), (1.0022028938495444, 0.0), (1.0012240625715843, 0.0), (1.0010304301156778, 0.0), (1.0016027123707445, 0.0), (1.001190603581833, 0.0), (1.000733777542572, 0.0), (1.001323717311767, 0.0), (1.0008946577069928, 0.0), (1.0003957574078266, 0.0), (1.0009011497885136, 0.0), (1.000995991086635, 0.0), (1.0010204705617347, 0.0), (1.0007385078217126, 0.0), (1.0009221489343825, 0.0), (1.0013036964829924, 0.0), (1.0013577689494564, 0.0), (1.0012290699754751, 0.0), (1.0009626047336173, 0.0), (1.0012406266835907, 0.0), (1.0013391038286579, 0.0)]
# BFV = [(0.500500001013279, 0.49949999898672104), (0.5009999945759773, 0.4990000054240227), (0.5016249939799309, 0.4983750060200691), (0.5023749992251396, 0.4976250007748604), (0.5032500103116035, 0.49674998968839645), (0.5042500123381615, 0.49574998766183853), (0.5053750202059746, 0.4946249797940254), (0.5066250339150429, 0.4933749660849571), (0.5080000385642052, 0.49199996143579483), (0.5095000490546227, 0.49049995094537735), (0.5111250653862953, 0.4888749346137047), (0.512875072658062, 0.487124927341938), (0.5147500857710838, 0.48524991422891617), (0.5167501047253609, 0.48324989527463913), (0.5188751146197319, 0.4811248853802681), (0.5211251303553581, 0.4788748696446419), (0.5235001519322395, 0.47649984806776047), (0.5260001644492149, 0.47399983555078506), (0.5286251828074455, 0.4713748171925545), (0.5313752070069313, 0.4686247929930687), (0.5342502221465111, 0.4657497778534889), (0.537250243127346, 0.46274975687265396), (0.5403752699494362, 0.4596247300505638), (0.5436252877116203, 0.45637471228837967), (0.5470003113150597, 0.45299968868494034), (0.5505003407597542, 0.4494996592402458), (0.5541253611445427, 0.4458746388554573), (0.5578753873705864, 0.4421246126294136), (0.5617504194378853, 0.4382495805621147), (0.5657504424452782, 0.43424955755472183), (0.5698754712939262, 0.43012452870607376), (0.5741255059838295, 0.4258744940161705), (0.5785005316138268, 0.42149946838617325), (0.5830005630850792, 0.4169994369149208), (0.5876256003975868, 0.4123743996024132), (0.5923756286501884, 0.40762437134981155), (0.5972506627440453, 0.40274933725595474), (0.6022507026791573, 0.39774929732084274), (0.6073757335543633, 0.39262426644563675), (0.6126257702708244, 0.38737422972917557), (0.6180008128285408, 0.3819991871714592), (0.6235008463263512, 0.37649915367364883), (0.6291258856654167, 0.3708741143345833), (0.6348759308457375, 0.36512406915426254), (0.6407509669661522, 0.3592490330338478), (0.6467510089278221, 0.3532489910721779), (0.6528760567307472, 0.3471239432692528), (0.6591260954737663, 0.3408739045262337), (0.6655011400580406, 0.3344988599419594), (0.6720011904835701, 0.3279988095164299), (0.6786262318491936, 0.3213737681508064), (0.6853762790560722, 0.31462372094392776), (0.6922513321042061, 0.3077486678957939), (0.6992513760924339, 0.30074862390756607), (0.706376425921917, 0.29362357407808304), (0.7136264815926552, 0.2863735184073448), (0.7210015282034874, 0.2789984717965126), (0.7285015806555748, 0.2714984193444252), (0.7361266389489174, 0.2638733610510826), (0.743876688182354, 0.256123311817646), (0.7517517432570457, 0.24824825674295425), (0.7597518041729927, 0.2402481958270073), (0.7678768560290337, 0.23212314397096634), (0.7761269137263298, 0.2238730862736702), (0.7845019772648811, 0.21549802273511887), (0.7930020317435265, 0.20699796825647354), (0.801627092063427, 0.19837290793657303), (0.8103771582245827, 0.18962284177541733), (0.8192522153258324, 0.18074778467416763), (0.8282522782683372, 0.17174772173166275), (0.8373773470520973, 0.16262265294790268), (0.8466274067759514, 0.15337259322404861), (0.8560024723410606, 0.14399752765893936), (0.8655025437474251, 0.13449745625257492), (0.8751276060938835, 0.12487239390611649), (0.8848776742815971, 0.11512232571840286), (0.894752748310566, 0.10524725168943405), (0.9047528132796288, 0.09524718672037125), (0.9148778840899467, 0.08512211591005325), (0.9251279607415199, 0.07487203925848007), (0.9355030283331871, 0.0644969716668129), (0.9460031017661095, 0.05399689823389053), (0.956628181040287, 0.04337181895971298), (0.9673782512545586, 0.03262174874544144), (0.9782533273100853, 0.021746672689914703), (0.9892534092068672, 0.010746590793132782), (1.0003784820437431, 0.0), (1.011256456375122, 0.0), (1.0113859176635742, 0.0), (1.0062555596232414, 0.0), (1.001146249473095, 0.0), (0.9986008778214455, 0.001399122178554535), (0.9971915781497955, 0.0028084218502044678), (0.99632678180933, 0.0036732181906700134), (0.995989017188549, 0.004010982811450958), (0.9961617067456245, 0.0038382932543754578), (0.9968291148543358, 0.003170885145664215), (0.9979762956500053, 0.0020237043499946594), (0.9995890632271767, 0.00041093677282333374), (1.0016539320349693, 0.0), (1.0025284588336945, 0.0), (1.0021887570619583, 0.0), (1.0014294311404228, 0.0), (1.0008459016680717, 0.0), (1.000639446079731, 0.0), (1.000719390809536, 0.0), (1.0008985549211502, 0.0), (1.001036398112774, 0.0), (1.0010852441191673, 0.0), (1.0010663866996765, 0.0), (1.0010240152478218, 0.0), (1.0009913817048073, 0.0), (1.0009798035025597, 0.0), (1.0009842664003372, 0.0)]

selected = CKKS
x = [x[0] for x in selected[-150:-10]]
y = [x[1] for x in selected[-150:-10]]

# x = [x[0] for x in selected]
# y = [x[1] for x in selected]

plt.scatter(x, y, linestyle='-', color='C2', label=r'CKKS')

# show the final plot
ax = plt.subplot(111)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_title(fr'Convergence pattern for CKKS')
ax.legend()
plt.xlim([.9999, 1.001])
plt.ylim([-.001, .004])
ax.grid('on')
plt.show()
