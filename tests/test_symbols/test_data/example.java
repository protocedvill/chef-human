import java.util.List;

public class Application {
    private String name;

    public Application(String name) {
        this.name = name;
    }

    public void run() {
        System.out.println("Running " + name);
    }

    public static int compute(int a, int b) {
        return a + b;
    }
}
