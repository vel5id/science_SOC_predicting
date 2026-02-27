
$refs = @(
    @{key="McBratney2003"; doi="10.1016/s0166-2481(06)31001-x"; expected="On digital soil mapping"; search="McBratney On digital soil mapping Geoderma 2003"},
    @{key="Wadoux2020"; doi="10.1016/j.geoderma.2020.114327"; expected="Machine learning for digital soil mapping review"; search="Wadoux Machine learning digital soil mapping review Geoderma 2020"},
    @{key="Behrens2018"; doi="10.1038/s41598-018-33516-6"; expected="Multi-scale digital soil mapping deep learning"; search="Behrens Multi-scale digital soil mapping deep learning Scientific Reports 2018"},
    @{key="Hengl2017"; doi="10.1371/journal.pone.0169748"; expected="SoilGrids250m"; search="Hengl SoilGrids250m PLoS ONE 2017"},
    @{key="Breiman2001"; doi="10.1023/A:1010933404324"; expected="Random forests"; search="Breiman Random forests Machine Learning 2001"},
    @{key="Chen2016"; doi="10.1145/2939672.2939785"; expected="XGBoost"; search="Chen XGBoost scalable tree boosting KDD 2016"},
    @{key="Dorogush2018"; doi="10.48550/arXiv.1706.09516"; expected="CatBoost"; search="Dorogush CatBoost gradient boosting 2018"},
    @{key="He2016"; doi="10.1109/cvpr.2016.90"; expected="Deep residual learning"; search="He Deep residual learning image recognition CVPR 2016"},
    @{key="Liu2022"; doi="10.1109/CVPR52688.2022.01167"; expected="ConvNet for the 2020s"; search="Liu ConvNet 2020s CVPR 2022"},
    @{key="Roberts2017"; doi="10.1111/ecog.02881"; expected="Cross-validation strategies"; search="Roberts Cross-validation strategies spatially structured Ecography 2017"},
    @{key="Geurts2006"; doi="10.1007/s10994-006-6226-1"; expected="Extremely randomized trees"; search="Geurts Extremely randomized trees Machine Learning 2006"},
    @{key="Smola2004"; doi="10.1023/B:STCO.0000035301.49549.88"; expected="tutorial support vector regression"; search="Smola tutorial support vector regression Statistics Computing 2004"},
    @{key="Padarian2019"; doi="10.5194/soil-5-79-2019"; expected="Using deep learning digital soil mapping"; search="Padarian Using deep learning digital soil mapping Soil 2019"},
    @{key="Castaldi2019"; doi="10.1016/j.isprsjprs.2018.11.026"; expected="Sentinel-2 soil organic carbon"; search="Castaldi Evaluating Sentinel-2 soil organic carbon ISPRS 2019"},
    @{key="Vaudour2019"; doi="10.1016/j.rse.2019.03.006"; expected="Sentinel-2 image capacities"; search="Vaudour Sentinel-2 image capacities topsoil properties Remote Sensing Environment 2019"},
    @{key="Haralick1973"; doi="10.1109/tsmc.1973.4309314"; expected="Textural features image classification"; search="Haralick Textural features image classification IEEE 1973"},
    @{key="Hu2018"; doi="10.1109/CVPR.2018.00745"; expected="Squeeze-and-excitation networks"; search="Hu Squeeze-and-excitation networks CVPR 2018"},
    @{key="MunozSabater2021"; doi="10.5194/essd-13-4349-2021"; expected="ERA5-Land"; search="Munoz-Sabater ERA5-Land ESSD 2021"},
    @{key="Drusch2012"; doi="10.1016/j.rse.2011.11.026"; expected="Sentinel-2"; search="Drusch Sentinel-2 multispectral Remote Sensing Environment 2012"},
    @{key="Torres2012"; doi="10.1016/j.rse.2012.02.014"; expected="GMES Sentinel-1"; search="Torres GMES Sentinel-1 mission Remote Sensing Environment 2012"},
    @{key="Pribyl2010"; doi="10.1016/j.geoderma.2010.02.003"; expected="SOC to SOM conversion"; search="Pribyl critical review conventional SOC SOM conversion Geoderma 2010"},
    @{key="Fick2017"; doi="10.1002/joc.5086"; expected="WorldClim 2"; search="Fick WorldClim 2 climate surfaces International Journal Climatology 2017"},
    @{key="Friedman2001"; doi="10.1214/aos/1013203451"; expected="Greedy function approximation"; search="Friedman Greedy function approximation gradient boosting Annals Statistics 2001"},
    @{key="Pedregosa2011"; doi="10.48550/arXiv.1201.0490"; expected="Scikit-learn"; search="Pedregosa Scikit-learn Machine Learning Python JMLR 2011"},
    @{key="Paszke2019"; doi="10.48550/arXiv.1912.01703"; expected="PyTorch"; search="Paszke PyTorch imperative style NeurIPS 2019"},
    @{key="Gebbers2010"; doi="10.1126/science.1183899"; expected="Precision agriculture food security"; search="Gebbers Precision agriculture food security Science 2010"},
    @{key="ViscarraRossel2006"; doi="10.1016/j.geoderma.2005.03.007"; expected="Visible NIR MIR spectroscopy"; search="Viscarra Rossel Visible NIR MIR spectroscopy soil Geoderma 2006"},
    @{key="Swinnen2017"; doi="10.1016/j.gfs.2017.03.005"; expected="Production potential bread baskets"; search="Swinnen Production potential bread baskets Global Food Security 2017"},
    @{key="Kraemer2015"; doi="10.1088/1748-9326/10/5/054012"; expected="Long-term agricultural land-cover"; search="Kraemer Long-term agricultural land-cover Environmental Research Letters 2015"},
    @{key="BenDor2009"; doi="10.1016/j.rse.2008.09.019"; expected="Using imaging spectroscopy"; search="Ben-Dor imaging spectroscopy soil Remote Sensing Environment 2009"},
    @{key="ViscarraRossel2010"; doi="10.1016/j.geoderma.2009.12.025"; expected="Using data mining"; search="Viscarra Rossel data mining soil Geoderma 2010"},
    @{key="Roy2014"; doi="10.1016/j.rse.2014.02.001"; expected="Landsat-8"; search="Roy Landsat-8 Remote Sensing Environment 2014"},
    @{key="Bauer2019"; doi="10.1109/tgrs.2018.2858004"; expected="Toward global soil moisture Sentinel-1"; search="Bauer global soil moisture monitoring Sentinel-1 IEEE TGRS 2019"},
    @{key="Farr2007"; doi="10.1029/2005RG000183"; expected="Shuttle Radar Topography Mission"; search="Farr Shuttle Radar Topography Mission Reviews Geophysics 2007"},
    @{key="Zizala2022"; doi="10.3390/rs14081941"; expected="Soil organic carbon mapping Sentinel-2"; search="Zizala Soil organic carbon mapping Sentinel-2 Remote Sensing 2022"},
    @{key="Wadoux2019"; doi="10.1016/j.geoderma.2019.05.012"; expected="Using deep learning multivariate mapping"; search="Wadoux deep learning multivariate mapping Geoderma 2019"},
    @{key="Wadoux2021"; doi="10.1016/j.ecolmodel.2021.109692"; expected="Spatial cross-validation not the right way"; search="Wadoux Spatial cross-validation not right way Ecological Modelling 2021"},
    @{key="Meyer2021"; doi="10.1111/2041-210X.13650"; expected="Predicting into unknown space"; search="Meyer Predicting into unknown space Methods Ecology Evolution 2021"},
    @{key="Gorelick2017"; doi="10.1016/j.rse.2017.06.031"; expected="Google Earth Engine"; search="Gorelick Google Earth Engine Remote Sensing Environment 2017"},
    @{key="Tucker1979"; doi="10.1016/0034-4257(79)90013-0"; expected="Red and photographic infrared"; search="Tucker Red photographic infrared Remote Sensing Environment 1979"},
    @{key="Hastie2009"; doi="10.1007/978-0-387-21606-5"; expected="Elements of Statistical Learning"; search="Hastie Elements Statistical Learning Springer 2009"},
    @{key="Hughes1968"; doi="10.1109/tit.1968.1054102"; expected="On the mean accuracy"; search="Hughes mean accuracy IEEE Transactions Information Theory 1968"},
    @{key="Zhong2024"; doi="MISSING"; expected="Soil properties prediction deep learning"; search="Zhong Soil properties prediction deep learning Geoderma 2024 116753"},
    @{key="Montanarella2016"; doi="10.5194/soil-2-79-2016"; expected="World soils are under threat"; search="Montanarella world soils are under threat SOIL 2016"},
    @{key="Bartholomeus2008"; doi="10.1016/j.geoderma.2008.01.010"; expected="Spectral reflectance based indices SOC"; search="Bartholomeus Spectral reflectance based indices soil organic carbon Geoderma 2008"},
    @{key="Grinsztajn2022"; doi="10.48550/arXiv.2207.08815"; expected="tree-based models outperform"; search="Grinsztajn tree-based models still outperform deep learning NeurIPS 2022"},
    @{key="Morellos2016"; doi="10.1016/j.biosystemseng.2016.04.018"; expected="Machine learning prediction soil properties"; search="Morellos Machine learning prediction soil properties Biosystems Engineering 2016"},
    @{key="Scherer2001"; doi="10.1016/s1161-0301(00)00082-4"; expected="Sulphur crop production"; search="Scherer Sulphur crop production European Journal Agronomy 2001"},
    @{key="Sims1996"; doi="10.2136/sssabookser4.2ed.c12"; expected="Micronutrient soil tests"; search="Sims Micronutrient soil tests SSSA 1996"},
    @{key="Wang2023SSL4EO"; doi="10.1109/MGRS.2023.3281651"; expected="SSL4EO-S12"; search="Wang SSL4EO-S12 IEEE Geoscience Remote Sensing Magazine 2023"}
)

$results = @()

foreach ($ref in $refs) {
    $status = ""
    $foundTitle = ""
    $correctDoi = ""

    if ($ref.doi -eq "MISSING") {
        $status = "MISSING"
    } else {
        # Verify DOI
        $encodedDoi = $ref.doi -replace '\(','%28' -replace '\)','%29'
        $url = "https://api.crossref.org/works/$encodedDoi"
        try {
            $r = Invoke-RestMethod -Uri $url -Method Get -TimeoutSec 20
            $foundTitle = ($r.message.title -join '; ').Trim()
            
            # Check if title roughly matches
            $expectedLower = $ref.expected.ToLower()
            $foundLower = $foundTitle.ToLower()
            
            $matchWords = ($expectedLower -split '\s+') | Where-Object { $foundLower -match [regex]::Escape($_) }
            $totalWords = ($expectedLower -split '\s+').Count
            
            if ($matchWords.Count -ge [math]::Ceiling($totalWords * 0.4)) {
                $status = "CORRECT"
            } else {
                $status = "WRONG_PAPER"
            }
        } catch {
            $status = "NOT_RESOLVED"
        }
    }

    Write-Output "=== $($ref.key) ==="
    Write-Output "  Current DOI: $($ref.doi)"
    Write-Output "  Status: $status"
    if ($foundTitle) { Write-Output "  Found title: $foundTitle" }
    Write-Output ""
    
    Start-Sleep -Milliseconds 400
}
