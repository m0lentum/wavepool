#include "../gfd/GFD/Mesh/BuilderMesh.hpp"
#include "../gfd/GFD/Output/MeshDrawer.hpp"
#include "../gfd/GFD/Types/Buffer.hpp"
#include "../gfd/GFD/Types/Types.hpp"
#include "../gfd/GFD/Types/Vector.hpp"
#include <functional>
#include <iostream>

using Vec2 = gfd::Vector2;
using Vec3 = gfd::Vector3;
using Vec4 = gfd::Vector4;

// numeerista integrointia varten
void discretise1Form(const gfd::Mesh &mesh, gfd::Buffer<double> &discreteForm,
                     const std::function<Vec2(Vec2)> &fn);

int main() {
  uint dimension = 2;
  double meshSize = 0.125; // verkon solujen suuruusluokka

  Vec2 p0(-1.0, -1.0); // verkon vasen alakulma
  Vec2 p1(1.0, 1.0);   // verkon oikea yl‰kulma

  // alustaa tyhj‰n verkon valittuun ulottuvuuteen
  gfd::BuilderMesh mesh(dimension);

  // luo verkon suorakulmioista
  mesh.createGrid(Vec4(p0, 0.0, 0.0), Vec4(p1, 0.0, 0.0), meshSize);

  // muitakin vaihtoehtoja voi kokeilla
  // mesh.createTriangleGrid(p0, p1, meshSize);
  // mesh.createHexagonGrid(p0, p1, meshSize);
  // mesh.createSnubSquareGrid(p0, p1, meshSize);
  // mesh.createTetrilleGrid(p0, p1, meshSize);

  // writeStatistics kertoo verkon ominaisuuksista
  gfd::Text text;
  mesh.writeStatistics(text);
  text.save("stat.txt");

  // MeshDrawer-luokkaa voi k‰ytt‰‰ verkon piirt‰miseen
  gfd::Picture pic(500, 500);
  gfd::MeshDrawer drawer;
  drawer.initPosition(Vec4(0, 0, 1, 0), Vec4(0, 0, 0, 0), Vec4(0.45, 0, 0, 0),
                      Vec4(0, 0.45, 0, 0));
  drawer.initPicture(&pic);
  drawer.drawPrimalEdges(mesh, Vec3(1.0, 1.0, 1.0));
  drawer.drawDualEdges(mesh, Vec3(0.5, 0.0, 0.0));
  pic.save("mesh.bmp", false);

  // tarkastellaan testifunktiota u(x,y) = (sin(pi*x)*(1-y^2), x*cos(pi*y))
  // (vastaa 1-muotoa u(x,y) = sin(pi*x)*(1-y^2) dx + x*cos(pi*y) dy)
  std::function<Vec2(Vec2)> u{[&](Vec2 p) -> Vec2 {
    return Vec2(sin(gfd::PI * p.x) * (1.0 - p.y * p.y),
                p.x * cos(gfd::PI * p.y));
  }};

  // approksimoidaan u:ta 1-cochainilla Cu (eli diskreetill‰ 1-muodolla)
  gfd::Buffer<double> Cu(mesh.getEdgeSize(), 0.0);
  // integroidaan u numeerisesti verkon s‰rmien yli
  discretise1Form(mesh, Cu, u);

  // kerrotaan Cu:ta matriisilla \star_1, jolloin saadaan 1-cochain
  // duaaliverkkoon
  gfd::Buffer<double> starCu(mesh.getEdgeSize());
  for (uint i = 0; i < mesh.getEdgeSize(); ++i) {
    starCu[i] =
        Cu[i] *
        mesh.getEdgeHodge(i); //<---diagonaalimatriisin \star_1 (i, i)-alkio
                              //(vastaavasti \star_0:lle getNodeHodge,
                              //\star_2:lle getFaceHodge jne, katso Mesh-luokka)
  }

  // kerrotaan starCu:ta diskreetin ulkoderivaatan d_0 transpoosilla, joka
  // toimii insidenssimatriisina duaaliverkossa,
  // saadaan 2-cochain duaaliverkkoon
  gfd::Buffer<double> d0tstarCu(mesh.getNodeSize(), 0.0);
  for (uint i = 0; i < mesh.getNodeSize(); ++i) {
    // vastaavasti lˆytyy getEdgeNodes, getFaceEdges jne, katso Mesh-luokka
    const gfd::Buffer<uint> &nodeEdges = mesh.getNodeEdges(i);
    for (uint e = 0; e < nodeEdges.size(); ++e) {
      uint j = nodeEdges[e];
      d0tstarCu[i] +=
          starCu[j] * mesh.getEdgeIncidence(
                          j, i); //<---matriisin d_0 (j, i)-alkio (vastaavasti
                                 // d_1:lle getFaceIncidence, katso Mesh-luokka)
    }
  }

  // kerrotaan -d0tstarCu:ta (huom. miinusmerkki) viel‰ matriisilla
  // \star_0^{-1}, jolloin saadaan 0-cochain alkuper‰iseen verkkoon
  gfd::Buffer<double> divu_approx(mesh.getNodeSize());
  for (uint i = 0; i < mesh.getNodeSize(); ++i) {
    divu_approx[i] = -d0tstarCu[i] / mesh.getNodeHodge(i);
  }

  // u:n oikea divergenssi on divu(x,y) = pi * (cos(pi*x)*(1-y^2) - x*sin(pi*y))
  std::function<double(Vec2)> divu{[&](Vec2 p) -> double {
    return gfd::PI *
           (cos(gfd::PI * p.x) * (1.0 - p.y * p.y) - p.x * sin(gfd::PI * p.y));
  }};

  // verrataan approksimaatiota oikeaan divergenssiin (verkon sis‰pisteiss‰
  // tulee kohtuullinen approksimaatio ainakin suorakulmio- ja kolmioverkoilla)
  mesh.fillBoundaryFlags(1); // merkitsee kaikki reunalla olevat solut (flag=1)
  for (uint i = 0; i < mesh.getNodeSize(); ++i) {
    if (mesh.getNodeFlag(i) == 1)
      continue;
    std::cout << "Noden indeksi: " << i << " Approksimaatio: " << divu_approx[i]
              << " Oikea arvo: " << divu(mesh.getNodePosition2(i)) << '\n';
  }
  std::cout << "\n\n\n";

  // harjoitusteht‰v‰: approksimoi DEC:ll‰ funktion v(x,y) = (x^2+1) * cos(pi*y)
  // laplaciaania ja vertaa approksimaatiota oikeaan laplaciaaniin lapv(x,y) =
  // (2 - pi^2(x^2+1)) * cos(pi*y)
  std::function<double(Vec2)> v{
      [&](Vec2 p) -> double { return (p.x * p.x + 1) * cos(gfd::PI * p.y); }};
  std::function<double(Vec2)> lapv{[&](Vec2 p) -> double {
    return (2 - gfd::PI * gfd::PI * (p.x * p.x + 1)) * cos(gfd::PI * p.y);
  }};

  // ratkaisu:

  // approksimoidaan v:t‰ 0-cochainilla Cv (eli diskreetill‰ 0-muodolla)
  gfd::Buffer<double> Cv(mesh.getNodeSize(), 0.0);
  //(t‰ydenn‰ ratkaisu loppuun)

  // 0-cochain on v:n arvo verkon solmupisteiss‰; ei tarvita integrointia
  for (uint i = 0; i < mesh.getNodeSize(); ++i) {
    Cv[i] = v(mesh.getNodePosition2(i));
  }

  // laplaciaani on gradientin divergenssi. Lasketaan ensin gradientti,
  // joka on 0-muodon ulkoderivaattaa vastaava 1-muoto.
  // 0-muodon ulkoderivaatta lasketaan v‰hent‰m‰ll‰ kunkin s‰rm‰n alkusolmun
  // arvo sen loppusolmun arvosta
  gfd::Buffer<double> gradCv(mesh.getEdgeSize(), 0.0);
  for (uint i_edge = 0; i_edge < mesh.getEdgeSize(); ++i_edge) {
    const gfd::Buffer<uint> &edgeEnds = mesh.getEdgeNodes(i_edge);
    // t‰ss‰ ei tarvita `mesh.getEdgeIncidence`‰, koska `mesh.getEdgeNodes`
    // palauttaa solmut s‰rm‰n orientaation mukaisessa j‰rjestyksess‰
    gradCv[i_edge] = Cv[edgeEnds[1]] - Cv[edgeEnds[0]];
  }

  // 3D:ss‰ divergenssi olisi 2-muodon ulkoderivaatta,
  // mutta 2D:ss‰ sellaista ei ole olemassa.
  // Divergenssi saadaan sen sijaan 90 astetta kierretyn 1-muodon
  // ulkoderivaattana (1-muodon ulkoderivaatta ilman kiertoa on roottori).
  // T‰m‰ saadaan aikaan muodostamalla ensin t‰htioperaattorilla 1-cochain
  // duaaliverkkoon ja sitten ottamalla sen ulkoderivaatta
  gfd::Buffer<double> starGradCv(mesh.getEdgeSize(), 0.0);
  for (uint i = 0; i < mesh.getEdgeSize(); ++i) {
    starGradCv[i] = gradCv[i] * mesh.getEdgeHodge(i);
  }

  // edellisen ulkoderivaatta on Cv:n laplaciaani 2-muotona duaaliverkossa.
  // T‰m‰ lasketaan samalla tavalla kuin aikaisemman esimerkin d0tstarCu
  gfd::Buffer<double> lapCvDual(mesh.getNodeSize(), 0.0);
  for (uint i_dface = 0; i_dface < mesh.getNodeSize(); ++i_dface) {
    const gfd::Buffer<uint> &dualFaceEdges = mesh.getNodeEdges(i_dface);
    for (uint e = 0; e < dualFaceEdges.size(); ++e) {
      uint i_edge = dualFaceEdges[e];
      lapCvDual[i_dface] +=
          starGradCv[i_edge] * mesh.getEdgeIncidence(i_edge, i_dface);
    }
  }

  // 0-cochain primaaliverkkoon saadaan kertomalla -1:ll‰
  // ja k‰‰nteisell‰ Hodge-t‰hdell‰
  gfd::Buffer<double> lapCvPrimal(mesh.getNodeSize(), 0.0);
  for (uint i = 0; i < mesh.getNodeSize(); ++i) {
    lapCvPrimal[i] = -lapCvDual[i] / mesh.getNodeHodge(i);
  }

  // vertailu
  for (uint i = 0; i < mesh.getNodeSize(); ++i) {
    if (mesh.getNodeFlag(i) == 1)
      continue;

    const double approx = lapCvPrimal[i];
    const double actual = lapv(mesh.getNodePosition2(i));
    const double error = abs(approx - actual);

    std::cout << "Node " << i << ", Approksimaatio: " << approx
              << ", Oikea arvo: " << actual << ", Virhe: " << error
              << std::endl;
  }
}

// numeerista integrointia varten
const std::vector<double> valueWeights1D{
    0.202578241925561272880620199967519314839,
    0.198431485327111576456118326443839324819,
    0.198431485327111576456118326443839324819,
    0.186161000015562211026800561866422824506,
    0.186161000015562211026800561866422824506,
    0.166269205816993933553200860481208811131,
    0.166269205816993933553200860481208811131,
    0.139570677926154314447804794511028322521,
    0.139570677926154314447804794511028322521,
    0.107159220467171935011869546685869303416,
    0.107159220467171935011869546685869303416,
    0.070366047488108124709267416450667338467,
    0.070366047488108124709267416450667338467,
    0.030753241996117268354628393577204417722,
    0.030753241996117268354628393577204417722};
const std::vector<double> abscissas{0.0,
                                    -0.20119409399743452230062830339459620781,
                                    0.20119409399743452230062830339459620781,
                                    -0.39415134707756336989720737098104546836,
                                    0.39415134707756336989720737098104546836,
                                    -0.57097217260853884753722673725391064124,
                                    0.57097217260853884753722673725391064124,
                                    -0.72441773136017004741618605461393800963,
                                    0.72441773136017004741618605461393800963,
                                    -0.84820658341042721620064832077421685137,
                                    0.84820658341042721620064832077421685137,
                                    -0.93727339240070590430775894771020947124,
                                    0.93727339240070590430775894771020947124,
                                    -0.987992518020485428489565718586612581147,
                                    0.987992518020485428489565718586612581147};

void discretise1Form(const gfd::Mesh &mesh, gfd::Buffer<double> &discreteForm,
                     const std::function<Vec2(Vec2)> &fn) {
  for (uint i = 0; i < mesh.getEdgeSize(); i++) {
    const gfd::Buffer<uint> &n = mesh.getEdgeNodes(i);
    Vec2 edgeVector = mesh.getNodePosition2(n[1]) - mesh.getNodePosition2(n[0]);
    Vec2 edgePosition = mesh.getNodePosition2(n[0]) + 0.5 * edgeVector;
    double sum = 0;
    for (uint j = 0; j < valueWeights1D.size(); ++j) {
      Vec2 evaluationPoint = edgePosition + abscissas[j] * 0.5 * edgeVector;
      sum += valueWeights1D[j] * edgeVector.dot(fn(evaluationPoint));
    }
    discreteForm[i] = 0.5 * sum;
  }
}
